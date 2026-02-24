# main.py
import os, shutil

# Use JDK 17 for PySpark 3.5 compatibility (getSubject removed in JDK 18+)
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
# Set Hadoop home for Windows winutils
os.environ["HADOOP_HOME"] = r"C:\hadoop"
# Clear any stale Java options
os.environ.pop("_JAVA_OPTIONS", None)

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("MusicAnalysis")
    .master("local[*]")
    .getOrCreate()
)

# ── Load datasets ───────────────────────────────────────────────────────────
logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
songs = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)

# Join logs with song metadata on song_id
joined = logs.join(songs, on="song_id", how="inner")

# Helper: write a single-partition CSV to an output folder
def save_csv(df, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    df.coalesce(1).write.csv(path, header=True, mode="overwrite")

# ── Task 1: User Favorite Genres ────────────────────────────────────────────
# For each user, find the genre they listened to the most.
genre_counts = joined.groupBy("user_id", "genre").agg(count("*").alias("listen_count"))

# Window: rank genres per user by listen_count descending
w1 = Window.partitionBy("user_id").orderBy(col("listen_count").desc())
user_fav_genre = (
    genre_counts
    .withColumn("rank", row_number().over(w1))
    .filter(col("rank") == 1)
    .select("user_id", "genre", "listen_count")
    .orderBy("user_id")
)

print("=== Task 1: User Favorite Genres ===")
user_fav_genre.show(10, truncate=False)
save_csv(user_fav_genre, "outputs/task1_user_favorite_genres")

# ── Task 2: Average Listen Time per Genre ────────────────────────────────────
avg_listen = (
    joined
    .groupBy("genre")
    .agg(round(avg("duration_sec"), 2).alias("avg_duration_sec"))
    .orderBy("genre")
)

print("=== Task 2: Average Listen Time per Genre ===")
avg_listen.show(truncate=False)
save_csv(avg_listen, "outputs/task2_avg_listen_time")

# ── Task 3: Genre Loyalty Scores — Top 10 ───────────────────────────────────
# Loyalty score = (plays of user's top genre) / (total plays by that user)
total_per_user = joined.groupBy("user_id").agg(count("*").alias("total_plays"))
top_genre_per_user = (
    genre_counts
    .withColumn("rank", row_number().over(w1))
    .filter(col("rank") == 1)
    .select("user_id", col("genre").alias("top_genre"), col("listen_count").alias("top_genre_plays"))
)

loyalty = (
    top_genre_per_user
    .join(total_per_user, on="user_id")
    .withColumn("loyalty_score", round(col("top_genre_plays") / col("total_plays"), 4))
    .orderBy(col("loyalty_score").desc())
    .limit(10)
    .select("user_id", "top_genre", "top_genre_plays", "total_plays", "loyalty_score")
)

print("=== Task 3: Genre Loyalty Scores — Top 10 ===")
loyalty.show(truncate=False)
save_csv(loyalty, "outputs/task3_genre_loyalty")

# ── Task 4: Night-owl Users (12 AM – 5 AM) ──────────────────────────────────
night_listeners = (
    joined
    .withColumn("hour", hour(col("timestamp")))
    .filter((col("hour") >= 0) & (col("hour") < 5))
    .select("user_id")
    .distinct()
    .orderBy("user_id")
)

print("=== Task 4: Night-Owl Users (12 AM – 5 AM) ===")
night_listeners.show(50, truncate=False)
save_csv(night_listeners, "outputs/task4_night_listeners")

spark.stop()
print("\nAll tasks complete. Check the outputs/ directory.")
