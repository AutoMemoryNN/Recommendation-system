import os
import re
import spacy.lang.en as en
import pandas as pd

from .CsvManager import CsvManager


class DataExtractor:
    """
    DataExtractor is responsible for extracting structured information from raw CSV files and create usefull CSV files for training.
    """

    output_path = "filtered-data"

    def coursera(self):
        """Extract the structured information from the Coursera.csv file"""
        csv_manager = CsvManager("raw-data/Coursera.csv")
        csv_manager.extractCoursera(f"{self.output_path}/Coursera.csv")

    def mediumBlogData(self):
        """Extract the structured information from the MediumBlogData.csv file"""
        csv_manager = CsvManager("raw-data/MediumBlogData.csv")
        csv_manager.extractMediumBlogData(f"{self.output_path}/MediumBlogData.csv")

    def posts(self):
        """Extract the structured information from the Posts.csv file"""
        csv_manager = CsvManager("raw-data/posts.csv")
        csv_manager.extractPost(f"{self.output_path}/posts.csv")

    def tedTalksEn(self):
        """Extract the structured information from the ted_talks_en.csv file"""
        csv_manager = CsvManager("raw-data/ted_talks_en.csv")
        csv_manager.extractTedTalksEn(f"{self.output_path}/ted_talks_en.csv")

    def udemyCourses(self):
        """Extract the structured information from the udemy_courses.csv file"""
        csv_manager = CsvManager("raw-data/udemy_courses.csv")
        csv_manager.extractUdemyCourses(f"{self.output_path}/udemy_courses.csv")

    def generateDataFile(self):
        """
        Generates a consolidated CSV file named 'data.csv' by merging the 'title' and 'topic'
        columns from all CSV files in the output directory, except 'data.csv' itself.
        The final dataset retains only the top 500 topics with the highest number of associated titles.
        """
        output_file = os.path.join(self.output_path, "data.csv")
        csv_files = [
            f
            for f in os.listdir(self.output_path)
            if f.endswith(".csv") and f != "data.csv"
        ]

        if not csv_files:
            raise ValueError("No valid CSV files found in the directory.")

        merged_data = []

        for file in csv_files:
            file_path = os.path.join(self.output_path, file)
            df = pd.read_csv(file_path, usecols=["title", "topic"], dtype=str)
            merged_data.append(df)

        result_df = pd.concat(merged_data, ignore_index=True).drop_duplicates()

        result_df = self.filterInput(result_df)

        # Count the number of titles per topic and keep the top 500 topics
        top_topics = result_df["topic"].value_counts().nlargest(500).index
        filtered_df = result_df[result_df["topic"].isin(top_topics)]

        filtered_df.to_csv(output_file, index=False)

        print(f"Generated: {output_file}")

    def filterInput(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame by performing the following transformations:

        1. Removes stop words from the 'title' column.
        2. Converts all words in 'title' to lowercase.
        3. Removes emojis from the 'title' column.
        4. Removes special characters from the 'title' column.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'title' column.

        Returns:
        pd.DataFrame: Processed DataFrame with cleaned 'title' values.
        """
        emoji_pattern = re.compile(
            """
            [\U0001f600-\U0001f64f]  # emoticons
            |[\U0001f300-\U0001f5ff]  # symbols & pictographs
            |[\U0001f680-\U0001f6ff]  # transport & map symbols
            |[\U0001f700-\U0001f77f]  # alchemical symbols
            |[\U0001f780-\U0001f7ff]  # Geometric Shapes Extended
            |[\U0001f800-\U0001f8ff]  # Supplemental Arrows-C
            |[\U0001f900-\U0001f9ff]  # Supplemental Symbols and Pictographs
            |[\U0001fa00-\U0001fa6f]  # Chess Symbols
            |[\U0001fa70-\U0001faff]  # Symbols and Pictographs Extended-A
            |[\U00002702-\U000027b0]  # Dingbats
            |[\U000024c2-\U0001f251]  # Enclosed characters
        """,
            re.VERBOSE,
        )

        special_chars_pattern = re.compile(r"[|!?<>:;\[\]{}=\-+_)(*&^%$#@!',]")

        df_title = df["title"].apply(
            lambda x: " ".join(
                [
                    re.sub(
                        special_chars_pattern,
                        "",
                        re.sub(emoji_pattern, "", word.lower()),
                    )
                    for word in x.split()
                    if word.lower() not in en.stop_words.STOP_WORDS
                ]
            )
        )

        df_title = df_title.apply(lambda x: x.replace("  ", " "))

        return pd.DataFrame({"title": df_title, "topic": df["topic"]})

    def printLabels(self, top=10):
        """
        Prints the unique labels in the 'topic' column of the 'data.csv' file.
        """

        # show each unique topic and also the nummer of members in each topic
        df = pd.read_csv(f"{self.output_path}/data.csv", dtype=str)

        for topic in df["topic"].value_counts().index[:top]:
            print(f"{topic}: {df[df['topic'] == topic].shape[0]}")


ds = DataExtractor()
# data extraction from raw data
# ds.coursera()
# ds.mediumBlogData()
# ds.posts()
# ds.tedTalksEn()
# ds.udemyCourses()

# generate consolidated data file
ds.generateDataFile()
# ds.printLabels(500)
