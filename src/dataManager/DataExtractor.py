from .CsvManager import CsvManager


class DataExtractor:
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


ds = DataExtractor()
# ds.coursera()
# ds.mediumBlogData()
# ds.posts()
# ds.tedTalksEn()
ds.udemyCourses()
