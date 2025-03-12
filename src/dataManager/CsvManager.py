import re
import pandas as pd
import json


class CsvManager:
    """
    CsvManager is responsible for handling raw CSV file operations such as reading from and writing to CSV files.
    """

    def __init__(self, file_path):
        """Initializes the manager with the CSV file."""
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def extractCoursera(self, output_path):
        """Converts the format of Skills and expands into multiple rows."""

        dfNew_rows = []

        for row in self.df.itertuples(index=False):
            try:
                courseName = row.CourseName.lower()
                skills = (
                    row.Skills.replace("[", "").replace("]", "").strip().split("  ")
                )

                for skill in skills:
                    skill = skill.lower().strip()
                    if skill:
                        dfNew_rows.append({"title": courseName, "topic": skill})

            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON en la fila {row}: {e}")

        dfNew = pd.DataFrame(dfNew_rows).drop_duplicates()

        print(
            f"Extracted Coursera: from {len(self.df)} rows to {len(dfNew)} rows after processing.."
        )

        dfNew.to_csv(output_path, index=False)

    def extractMediumBlogData(self, output_path):
        """Extracts the structured information from the MediumBlogData.csv file."""
        dfNew_rows = []

        for row in self.df.itertuples(index=False):
            title = row.blog_title
            topic = str(row.topic).replace("-", " ").strip()

            dfNew_rows.append({"title": title, "topic": topic})

        dfNew = pd.DataFrame(dfNew_rows)
        dfNew.to_csv(output_path, index=False)

    def extractPost(self, output_path):
        """Extracts the structured information from the Post.csv file."""
        dfNew_rows = []

        for row in self.df.itertuples(index=False):
            title = row.title
            topics = str(row.category).strip().split("|")

            for topic in topics:
                if topic:
                    dfNew_rows.append({"title": title, "topic": topic})

        dfNew = pd.DataFrame(dfNew_rows).drop_duplicates()
        print(
            f"Medium Posts: from {len(self.df)} rows to {len(dfNew)} rows after processing."
        )
        dfNew.to_csv(output_path, index=False)

    def extractTedTalksEn(self, output_path):
        """Extracts the structured information from the ted_talks_en.csv file."""
        dfNew_rows = []

        for row in self.df.itertuples(index=False):
            title = row.title
            topics = (
                str(row.topics)
                .replace("'", "")
                .replace("[", "")
                .replace("]", "")
                .split(",")
            )

            for topic in topics:
                if topic:
                    dfNew_rows.append({"title": title, "topic": topic.strip()})

        dfNew = pd.DataFrame(dfNew_rows).drop_duplicates()
        print(
            f"Extracted TED Talks: from {len(self.df)} rows to {len(dfNew)} rows after processing."
        )
        dfNew.to_csv(output_path, index=False)

    def extractUdemyCourses(self, output_path):
        """Extracts the structured information from the udemy_courses.csv file."""
        dfNew_rows = []
        skipped_titles = []
        skip_count = 0

        for row in self.df.itertuples(index=False):
            title = row.course_title

            if not re.fullmatch(
                r"[A-Za-z0-9\u00C0-\u024F\s\.,!?;:'\"()&/\+\|\#\-\u2014]+", title
            ):
                # Guardar solo los primeros 5 títulos omitidos
                if skip_count < 5:
                    skipped_titles.append(title)
                skip_count += 1
                continue

            topic = str(row.subject).strip()
            if topic:
                dfNew_rows.append({"title": title, "topic": topic})

        dfNew = pd.DataFrame(dfNew_rows).drop_duplicates()
        print(
            f"Udemy Courses: Processed {len(self.df)} rows, retained {len(dfNew)} rows."
        )
        if skip_count > 0:
            sample_skips = ", ".join(skipped_titles)
            print(
                f"Skipped {skip_count} titles. Examples of skipped titles: {sample_skips}"
            )
        dfNew.to_csv(output_path, index=False)
        print(f"Consolidated Udemy courses saved to: {output_path}")

    def save_to_csv(self, output_path):
        """Guarda el DataFrame modificado en un archivo CSV."""
        self.df.to_csv(output_path, index=False)

    def display(self, n=5):
        """Muestra las primeras 'n' filas del DataFrame."""
        print(self.df.head(n))

    def remove_columns(self, column_names):
        """Removes the specified columns."""
        self.df.drop(columns=column_names, inplace=True, errors="ignore")

    def remove_duplicates(self):
        """Removes duplicate rows from the DataFrame."""
        self.df.drop_duplicates(inplace=True)

    def save_to_csv(self, output_path=None):
        """Saves the modified DataFrame to a CSV file."""
        path = output_path if output_path else self.file_path
        self.df.to_csv(path, index=False)

    def display(self, n=5):
        """Displays the first 'n' rows of the DataFrame."""
        print(self.df.head(n))
