import os
import re
import spacy.lang.en as en
from transformers import BertTokenizer
import pandas as pd

from .CsvManager import CsvManager


class DataExtractor:
    """
    DataExtractor is responsible for extracting structured information from CSV files
    and creating useful CSV files for model training.
    """

    # Directory configuration
    RAW_DATA_DIR = "raw-data"
    FILTERED_DATA_DIR = "filtered-data"
    NON_TOKENIZED_DIR = "filtered-data/non-tokenized"
    TOKENIZED_DATA_DIR = "data"

    # Hyperparameter configuration
    DEFAULT_TOP_TOPICS = 400
    DEFAULT_VAL_RATIO = 0.1
    DEFAULT_TEST_RATIO = 0.1
    RANDOM_SEED = 43

    def __init__(self):
        """Initializes the data extractor and creates the necessary directories."""
        self._create_directories()

    def _create_directories(self):
        """Creates the necessary directories if they do not exist."""
        os.makedirs(self.FILTERED_DATA_DIR, exist_ok=True)
        os.makedirs(self.NON_TOKENIZED_DIR, exist_ok=True)
        os.makedirs(self.TOKENIZED_DATA_DIR, exist_ok=True)

    def extract_all_datasets(self):
        """Extracts all available datasets."""
        print("Step 1: Extracting data from all sources...")
        self.extract_coursera()
        self.extract_medium_blog_data()
        self.extract_posts()
        self.extract_ted_talks_en()
        self.extract_udemy_courses()
        print("Data extraction completed.")

    def extract_coursera(self):
        """Extracts structured information from the Coursera.csv file"""
        print("Extracting data from Coursera...")
        csv_manager = CsvManager(f"{self.RAW_DATA_DIR}/Coursera.csv")
        csv_manager.extractCoursera(f"{self.FILTERED_DATA_DIR}/Coursera.csv")

    def extract_medium_blog_data(self):
        """Extracts structured information from the MediumBlogData.csv file"""
        print("Extracting data from Medium Blog...")
        csv_manager = CsvManager(f"{self.RAW_DATA_DIR}/MediumBlogData.csv")
        csv_manager.extractMediumBlogData(
            f"{self.FILTERED_DATA_DIR}/MediumBlogData.csv"
        )

    def extract_posts(self):
        """Extracts structured information from the Posts.csv file"""
        print("Extracting data from Posts...")
        csv_manager = CsvManager(f"{self.RAW_DATA_DIR}/posts.csv")
        csv_manager.extractPost(f"{self.FILTERED_DATA_DIR}/posts.csv")

    def extract_ted_talks_en(self):
        """Extracts structured information from the ted_talks_en.csv file"""
        print("Extracting data from TED Talks...")
        csv_manager = CsvManager(f"{self.RAW_DATA_DIR}/ted_talks_en.csv")
        csv_manager.extractTedTalksEn(f"{self.FILTERED_DATA_DIR}/ted_talks_en.csv")

    def extract_udemy_courses(self):
        """Extracts structured information from the udemy_courses.csv file"""
        print("Extracting data from Udemy Courses...")
        csv_manager = CsvManager(f"{self.RAW_DATA_DIR}/udemy_courses.csv")
        csv_manager.extractUdemyCourses(f"{self.FILTERED_DATA_DIR}/udemy_courses.csv")

    def generate_consolidated_data(self, n_top_topics: int = None):
        """
        Generates a consolidated CSV file 'data.csv' by combining the 'title' and 'topic' columns
        from all CSV files in the output directory, except 'data.csv'.
        The final dataset retains only the top topics with the highest number of associated titles.

        Parameters:
        n_top_topics (int, optional): The number of top topics to retain. If None, uses the default value.
        """
        if n_top_topics is None:
            n_top_topics = self.DEFAULT_TOP_TOPICS

        print(
            f"\nStep 2: Generating consolidated data file with the top {n_top_topics} topics..."
        )

        output_file = os.path.join(self.FILTERED_DATA_DIR, "data.csv")
        csv_files = [
            f
            for f in os.listdir(self.FILTERED_DATA_DIR)
            if f.endswith(".csv") and f != "data.csv"
        ]

        if not csv_files:
            raise ValueError("No valid CSV files found in the directory.")

        merged_data = []

        for file in csv_files:
            file_path = os.path.join(self.FILTERED_DATA_DIR, file)
            df = pd.read_csv(file_path, usecols=["title", "topic"], dtype=str)
            merged_data.append(df)

        result_df = pd.concat(merged_data, ignore_index=True).drop_duplicates()
        result_df = self._filter_input(result_df)

        # Count the number of titles per topic and save the top n topics
        top_topics = result_df["topic"].value_counts().nlargest(n_top_topics).index
        filtered_df = result_df[result_df["topic"].isin(top_topics)]

        filtered_df.to_csv(output_file, index=False)
        print(f"Consolidated file generated: {output_file}")

        return filtered_df

    def _filter_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame by performing the following transformations:

        1. Removes stop words from the 'title' column.
        2. Converts all words in 'title' to lowercase.
        3. Removes emojis from the 'title' column.
        4. Removes special characters from the 'title' column.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'title' column.

        Returns:
        pd.DataFrame: Processed DataFrame with clean 'title' values.
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

        df_title = df_title.apply(
            lambda x: x.replace("  ", " ")
            .replace("'", "")
            .replace('"', "")
            .replace(".", "")
            .replace("/", " ")
            .replace("-", " ")
        )

        return pd.DataFrame({"title": df_title, "topic": df["topic"]})

    def print_topic_distribution(self, top_n: int = 10):
        """
        Displays the distribution of topics in the 'data.csv' file.

        Parameters:
        top_n (int): Number of top topics to display.
        """
        print(f"\nDistribution of the top {top_n} topics:")

        try:
            df = pd.read_csv(f"{self.FILTERED_DATA_DIR}/data.csv", dtype=str)
            topic_counts = df["topic"].value_counts()

            for topic in topic_counts.index[:top_n]:
                print(f"{topic}: {topic_counts[topic]}")

            print(f"\nTotal topics: {len(topic_counts)}")
            print(f"Total examples: {len(df)}")
        except FileNotFoundError:
            print(
                "The file data.csv does not exist. Please run generate_consolidated_data() first."
            )

    def split_and_tokenize_data(
        self,
        val_ratio: float = None,
        test_ratio: float = None,
        tokenizer_name: str = "bert-base-uncased",
    ):
        """
        Splits the consolidated data into training, validation, and test sets,
        and tokenizes the data for model training.

        Parameters:
        val_ratio (float, optional): Validation set ratio.
        test_ratio (float, optional): Test set ratio.
        tokenizer_name (str): Name of the tokenizer to use.
        """
        if val_ratio is None:
            val_ratio = self.DEFAULT_VAL_RATIO
        if test_ratio is None:
            test_ratio = self.DEFAULT_TEST_RATIO
            print(
                f"\nStep 3: Splitting and tokenizing data (val_ratio={val_ratio}, test_ratio={test_ratio})..."
            )

            try:
                df = pd.read_csv(f"{self.FILTERED_DATA_DIR}/data.csv", dtype=str)
            except FileNotFoundError:
                print(
                    "The file data.csv does not exist. Please run generate_consolidated_data() first."
                )
                return

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for topic, topic_df in df.groupby("topic"):
            total_samples = len(topic_df)
            val_samples = int(total_samples * val_ratio)
            test_samples = int(total_samples * test_ratio)
            train_samples = total_samples - val_samples - test_samples

            # Shuffle randomly
            topic_df_shuffled = topic_df.sample(frac=1, random_state=self.RANDOM_SEED)

            train_split = topic_df_shuffled.iloc[:train_samples]
            val_split = topic_df_shuffled.iloc[
                train_samples : train_samples + val_samples
            ]
            test_split = topic_df_shuffled.iloc[train_samples + val_samples :]

            train_dfs.append(train_split)
            val_dfs.append(val_split)
            test_dfs.append(test_split)

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Save non-tokenized versions
        train_path = f"{self.NON_TOKENIZED_DIR}/train.csv"
        val_path = f"{self.NON_TOKENIZED_DIR}/val.csv"
        test_path = f"{self.NON_TOKENIZED_DIR}/test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("Non-tokenized datasets saved.")
        print("Tokenizing data...")

        # Tokenize and save
        self._tokenize_data(
            test_df, f"{self.TOKENIZED_DATA_DIR}/test.csv", tokenizer_name
        )
        self._tokenize_data(
            val_df, f"{self.TOKENIZED_DATA_DIR}/val.csv", tokenizer_name
        )
        self._tokenize_data(
            train_df, f"{self.TOKENIZED_DATA_DIR}/train.csv", tokenizer_name
        )

        print("\nData split statistics:")
        print(f"Training set: {len(train_df)} examples")
        print(f"Validation set: {len(val_df)} examples")
        print(f"Test set: {len(test_df)} examples")

        print("\nTopic distribution:")
        print("Training topics:")
        print(train_df["topic"].value_counts().head())
        print("\nValidation topics:")
        print(val_df["topic"].value_counts().head())
        print("\nTest topics:")
        print(test_df["topic"].value_counts().head())

    def _tokenize_data(self, df: pd.DataFrame, output_path: str, tokenizer_name: str):
        """
        Tokenizes the 'title' column using the specified tokenizer.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'title' column.
        output_path (str): Path to save the tokenized data.
        tokenizer_name (str): Name of the tokenizer to use.
        """
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        df_title = df["title"].apply(
            lambda x: tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(str(x).lower())
            )
        )

        df_output = pd.DataFrame({"title": df_title, "topic": df["topic"]})
        df_output.to_csv(output_path, index=False)

        print(f"Tokenized data saved to: {output_path}")

    def process_pipeline(
        self,
        n_top_topics: int = None,
        val_ratio: float = None,
        test_ratio: float = None,
    ):
        """
        Runs the complete data processing pipeline in a single method.

        Parameters:
        n_top_topics (int, optional): Number of top topics to keep.
        val_ratio (float, optional): Validation dataset ratio.
        test_ratio (float, optional): Test dataset ratio.
        """
        print("\nStarting full data processing pipeline...")

        # Step 1: Extract raw data
        self.extract_all_datasets()

        # Step 2: Generate consolidated data file
        self.generate_consolidated_data(n_top_topics)

        # Step 3: Split and tokenize data
        self.split_and_tokenize_data(val_ratio, test_ratio)

        print("\nData processing pipeline completed successfully!")


de = DataExtractor()
de.process_pipeline()
