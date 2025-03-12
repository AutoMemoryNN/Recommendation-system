import ast
import json
import os
import re
import numpy as np
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
    TOKENIZED_DATA_DIR = "data/csv_data"
    ONE_HOT_MAPPING_DIR = "data"
    NP_X_DATA_PATH = "data/x_data"
    NP_Y_DATA_PATH = "data/y_data"

    SKIP_TOPICS = []

    # Hyperparameter configuration
    DEFAULT_TOP_TOPICS = 400
    DEFAULT_VAL_RATIO = 0.1
    DEFAULT_TEST_RATIO = 0.1
    RANDOM_SEED = 43

    ADD_PADDING = True

    LEN_OF_TOKENS = 0

    def __init__(self):
        """Initializes the data extractor and creates the necessary directories."""
        self._create_directories()

    def _create_directories(self):
        """Creates the necessary directories if they do not exist."""
        os.makedirs(self.FILTERED_DATA_DIR, exist_ok=True)
        os.makedirs(self.NON_TOKENIZED_DIR, exist_ok=True)
        os.makedirs(self.TOKENIZED_DATA_DIR, exist_ok=True)
        os.makedirs(self.NP_X_DATA_PATH, exist_ok=True)
        os.makedirs(self.NP_Y_DATA_PATH, exist_ok=True)
        os.makedirs(self.ONE_HOT_MAPPING_DIR, exist_ok=True)

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
        from all CSV files in the filtered data directory (excluding 'data.csv').
        Only the top topics (with the highest number of associated titles) are retained.

        Parameters:
        n_top_topics (int, optional): Number of top topics to retain. If None, uses the default value.

        Returns:
        pd.DataFrame: The filtered DataFrame containing only the top topics.
        """
        if n_top_topics is None:
            n_top_topics = self.DEFAULT_TOP_TOPICS

        print("\n[Step 2] Consolidated Data Generation")
        print(f" - Retaining the top {n_top_topics} topics based on title frequency.")

        output_file = os.path.join(self.FILTERED_DATA_DIR, "data.csv")
        csv_files = [
            f
            for f in os.listdir(self.FILTERED_DATA_DIR)
            if f.endswith(".csv") and f != "data.csv"
        ]

        if not csv_files:
            raise ValueError("ERROR: No valid CSV files found in the directory.")

        merged_data = []
        print("\n[Info] Reading and merging CSV files:")
        for file in csv_files:
            file_path = os.path.join(self.FILTERED_DATA_DIR, file)
            print(f" - Reading file: {file_path}")
            df = pd.read_csv(file_path, usecols=["title", "topic"], dtype=str)
            merged_data.append(df)

        result_df = pd.concat(merged_data, ignore_index=True).drop_duplicates()
        print(
            f"\n[Info] Total merged samples (after removing duplicates): {len(result_df)}"
        )

        # Aplicar filtro a la columna 'title'
        result_df = self._filter_input(result_df)

        # Seleccionar los tópicos más frecuentes
        top_topics = result_df["topic"].value_counts().nlargest(n_top_topics).index
        filtered_df = result_df[result_df["topic"].isin(top_topics)]
        print(
            f"[Info] Retained topics count: {len(filtered_df)} samples out of {len(result_df)}"
        )

        filtered_df.to_csv(output_file, index=False)
        print(f"\n[Success] Consolidated file generated and saved at: {output_file}")

        return filtered_df

    def filter_text(self, text: str) -> str:
        """
        Filters a text string by performing the following transformations:

        1. Removes stop words
        2. Converts all words to lowercase
        3. Removes emojis
        4. Removes special characters

        Parameters:
        text (str): Input text to filter

        Returns:
        str: Filtered text
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

        special_chars_pattern = re.compile(r"[|!?<>:;\[\]{}=\-+_)(*&^$#@!',]")

        # Process each word in the text
        filtered_text = " ".join(
            [
                re.sub(
                    special_chars_pattern,
                    "",
                    re.sub(emoji_pattern, "", word.lower()),
                )
                for word in text.split()
                if word.lower() not in en.stop_words.STOP_WORDS
            ]
        )

        # Additional cleanup
        filtered_text = (
            filtered_text.replace("  ", " ")
            .replace("'", "")
            .replace('"', "")
            .replace(".", "")
            .replace("/", " ")
            .replace("-", " ")
        )

        return filtered_text

    def _filter_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame by filtering the 'title' column and removing rows
        with topics listed in self.SKIP_TOPICS.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing a 'title' column.

        Returns:
            pd.DataFrame: Processed DataFrame with cleaned 'title' values and filtered topics.
        """
        print("\n[Info] Rows before filtering SKIP_TOPICS:", len(df))

        df_filtered = df[~df["topic"].isin(self.SKIP_TOPICS)]

        print("[Info] Rows after filtering SKIP_TOPICS:", len(df_filtered))

        df_title = df_filtered["title"].apply(self.filter_text)

        return pd.DataFrame({"title": df_title, "topic": df_filtered["topic"]})

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
            f"\n[Step 3] Splitting and tokenizing data\n"
            f" - Validation ratio: {val_ratio}\n"
            f" - Test ratio: {test_ratio}"
        )

        try:
            df = pd.read_csv(f"{self.FILTERED_DATA_DIR}/data.csv", dtype=str)
        except FileNotFoundError:
            print(
                "ERROR: The file 'data.csv' does not exist. Please run generate_consolidated_data() first."
            )
            return

        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Procesar cada grupo de datos por tópico
        for topic, topic_df in df.groupby("topic"):
            total_samples = len(topic_df)
            val_samples = int(total_samples * val_ratio)
            test_samples = int(total_samples * test_ratio)
            train_samples = total_samples - val_samples - test_samples

            # Mezclar aleatoriamente los datos del tópico
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

        # Guardar versiones sin tokenizar
        non_token_train_path = f"{self.NON_TOKENIZED_DIR}/train.csv"
        non_token_val_path = f"{self.NON_TOKENIZED_DIR}/val.csv"
        non_token_test_path = f"{self.NON_TOKENIZED_DIR}/test.csv"

        train_df.to_csv(non_token_train_path, index=False)
        val_df.to_csv(non_token_val_path, index=False)
        test_df.to_csv(non_token_test_path, index=False)

        print("\n[Info] Non-tokenized datasets saved to:")
        print(f"   Train: {non_token_train_path}")
        print(f"   Validation: {non_token_val_path}")
        print(f"   Test: {non_token_test_path}")

        print("\n[Info] Starting tokenization process...")
        train_path = f"{self.TOKENIZED_DATA_DIR}/train.csv"
        val_path = f"{self.TOKENIZED_DATA_DIR}/val.csv"
        test_path = f"{self.TOKENIZED_DATA_DIR}/test.csv"

        # Tokenizar y guardar
        self._tokenize_data(val_df, val_path, tokenizer_name)
        self._tokenize_data(test_df, test_path, tokenizer_name)
        self._tokenize_data(train_df, train_path, tokenizer_name)

        if self.ADD_PADDING:
            print("\n[Info] Applying padding to tokenized datasets...\n")
            self.add_padding(val_path)
            self.add_padding(test_path)
            self.add_padding(train_path)

        print("\n=== Data Split Statistics ===")
        print(f"Training set: {len(train_df)} examples")
        print(f"Validation set: {len(val_df)} examples")
        print(f"Test set: {len(test_df)} examples")

        print("\n=== Top 5 Topics Distribution ===")
        print("Training topics:")
        print(train_df["topic"].value_counts().head())

    def tokenize_text(
        self, text: str, tokenizer: BertTokenizer.from_pretrained = None
    ) -> list:
        """
        Tokenizes a single text string using the provided tokenizer.

        Parameters:
        text (str): The text to tokenize.
        tokenizer: The tokenizer object to use.

        Returns:
        list: The tokenized text as a list of token IDs.
        """
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(text).lower()))

    def _tokenize_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        tokenizer_name: str = "bert-base-uncased",
    ):
        """
        Tokenizes the 'title' column using the specified tokenizer and saves to a file.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'title' column.
        output_path (str): Path to save the tokenized data.
        tokenizer_name (str): Name of the tokenizer to use (default: "bert-base-uncased").
        """
        print(f"[Tokenization] Initializing tokenizer '{tokenizer_name}'...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Tokenize each title using the tokenize_text function
        df_title = df["title"].apply(lambda x: self.tokenize_text(x, tokenizer))
        print("[Tokenization] Tokenization of 'title' column completed.")

        # Create output DataFrame and remove rows with empty tokenized titles
        df_output = pd.DataFrame({"title": df_title, "topic": df["topic"]})
        initial_rows = len(df_output)
        df_output = df_output[df_output["title"].apply(lambda x: len(x) > 0)]
        removed = initial_rows - len(df_output)
        print(f"[Cleaning] Removed {removed} rows with empty tokenized titles.")

        # Si se requiere padding, calcular estadísticas de longitud
        if self.ADD_PADDING:
            lengths = df_output["title"].apply(len).tolist()
            max_length = max(lengths)
            min_length = min(lengths)
            avg_length = sum(lengths) / len(lengths) if lengths else 0

            if max_length > self.LEN_OF_TOKENS:
                self.LEN_OF_TOKENS = max_length

            print(
                f"[Padding] Token lengths statistics: Max = {max_length}, Min = {min_length}, Avg = {avg_length:.2f}"
            )

        df_output.to_csv(output_path, index=False)
        print(f"[Save] Tokenized data successfully saved to: {output_path}\n")

    def pad_token(self, token_list: list) -> list:
        """
        Adds padding to a token sequence.

        Parameters:
        token_list (list): List of tokens to pad

        Returns:
        list: Padded token list
        """
        if self.LEN_OF_TOKENS == 0:
            raise ValueError("ERROR: LEN_OF_TOKENS is not set. Please set it first.")
        # Add padding
        padded_tokens = token_list + [0] * (self.LEN_OF_TOKENS - len(token_list))
        return padded_tokens

    def add_padding(self, path: str):
        """
        Adds padding to the tokenized data in a file.

        Parameters:
        path (str): Path to the tokenized data file to add padding.
        """
        df = pd.read_csv(path, dtype=str)
        padded_titles = []

        for title_str in df["title"]:
            # Convert string representation of list to actual list of integers
            title_tokens = eval(title_str)
            if not isinstance(title_tokens, list):
                title_tokens = [int(title_str)] if title_str.isdigit() else []

            padded_tokens = self.pad_token(title_tokens)

            # Convert back to string representation
            padded_titles.append(str(padded_tokens))

        df_output = pd.DataFrame({"title": padded_titles, "topic": df["topic"]})
        df_output.to_csv(path, index=False)

    def save_data_to_numpy(self):
        """
        Save data to numpy arrays and perform necessary transformations.
        This method performs the following steps:
        1. Loads CSV files for training, validation, and testing datasets.
        2. Converts string representations of lists (e.g., "[1, 2, 3, 4, 5]") into numpy arrays.
        3. Transforms the 'title' column lists into numpy arrays.
        4. Saves the tokenized data into .npy files.
        5. One-hot encodes the 'topic' labels and saves them.
        Prints informative messages at each step to indicate progress.
        Raises:
            FileNotFoundError: If any of the CSV files are not found.
            ValueError: If there is an issue with converting string representations to numpy arrays.
        """
        print("\n[Step 4] Saving data to numpy arrays...\n")

        csv_test_path = f"{self.TOKENIZED_DATA_DIR}/test.csv"
        csv_val_path = f"{self.TOKENIZED_DATA_DIR}/val.csv"
        csv_train_path = f"{self.TOKENIZED_DATA_DIR}/train.csv"

        print("[INFO] Loading CSV files...")
        test_data = pd.read_csv(csv_test_path)
        val_data = pd.read_csv(csv_val_path)
        train_data = pd.read_csv(csv_train_path)
        print(
            f"[INFO] Loaded datasets: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)})"
        )

        # Convert string representations (e.g., "[1, 2, 3, 4, 5]") into numpy arrays
        print("[INFO] Converting string-based vectors to numpy arrays...")
        for df_name, df in zip(
            ["Test", "Validation", "Train"], [test_data, val_data, train_data]
        ):
            for col in df.columns:
                if df[col].dtype == "object" and not df[col].empty:
                    first_val = df[col].iloc[0]
                    if (
                        isinstance(first_val, str)
                        and first_val.strip().startswith("[")
                        and first_val.strip().endswith("]")
                    ):
                        df[col] = df[col].apply(
                            lambda elem: np.array(ast.literal_eval(elem))
                        )
            print(f"[INFO] {df_name} dataset processed.")

        # Convert the 'title' column lists into numpy arrays
        print("[INFO] Transforming 'title' columns into numpy matrices...")
        X_train = np.array(train_data["title"].tolist())
        X_val = np.array(val_data["title"].tolist())
        X_test = np.array(test_data["title"].tolist())
        print(
            f"[INFO] Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        # Save tokenized data
        np.save(self.NP_X_DATA_PATH + "/train", X_train)
        np.save(self.NP_X_DATA_PATH + "/val", X_val)
        np.save(self.NP_X_DATA_PATH + "/test", X_test)
        print("[INFO] Tokenized data saved.")

        # One-hot encode and save topic labels
        print("[INFO] One-hot encoding topics...")
        self._oneHotEncode(train_data["topic"].tolist(), self.NP_Y_DATA_PATH + "/train")
        self._oneHotEncode(val_data["topic"].tolist(), self.NP_Y_DATA_PATH + "/val")
        self._oneHotEncode(test_data["topic"].tolist(), self.NP_Y_DATA_PATH + "/test")
        print("[INFO] One-hot encoding completed.")

    def _oneHotEncode(self, targets: list, save_path: str):
        """
        One-hot encodes a list of target labels and saves the mapping as a JSON file.

        Parameters:
            targets (list): List of target labels to be one-hot encoded.
            save_path (str): Path where the one-hot encoded numpy array would be saved
                            (if needed) and the mapping will be stored in ONE_HOT_MAPPING_DIR.
        """
        unique_labels = sorted(list(set(targets)))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        n_samples = len(targets)
        n_classes = len(unique_labels)

        print(f"[INFO] Encoding {n_samples} labels into {n_classes} categories.")

        # Create empty array of shape (n_samples, n_classes)
        one_hot_encoded = np.zeros((n_samples, n_classes))
        for i, label in enumerate(targets):
            one_hot_encoded[i, label_to_index[label]] = 1

        # Save the mapping as JSON
        mapping_path = self.ONE_HOT_MAPPING_DIR + "/one_hot_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(label_to_index, f)
        print(f"[INFO] Saved label mapping to {mapping_path}")

        # Optionally, save the one-hot encoded array if needed:
        # np.save(save_path, one_hot_encoded)

    def onehot_to_string(self, onehot: np.ndarray):
        """
        Takes a one-hot encoded array or an index and returns the string representation of the label.
        Loads the mapping from ONE_HOT_MAPPING_DIR/one_hot_mapping.json to get the mapping.

        Parameters:
            onehot (np.ndarray or int): One-hot encoded array or index.

        Returns:
            str: The label corresponding to the given index.
        """
        mapping_path = self.ONE_HOT_MAPPING_DIR + "/one_hot_mapping.json"
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        # Reverse the mapping: keys become indices, values become topics.
        index_to_label = {v: k for k, v in mapping.items()}

        # If the input is a numpy array, convert it to int.
        if isinstance(onehot, np.ndarray):
            onehot = int(onehot)  # or onehot.item() if it's a single element array

        return index_to_label[onehot]

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

        # Step 4: Save data to numpy
        self.save_data_to_numpy()

        print("\nData processing pipeline completed successfully!")


# de = DataExtractor()
# de.SKIP_TOPICS = [".properties"]
# de.process_pipeline()
