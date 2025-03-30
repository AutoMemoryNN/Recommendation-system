from dataManager.DataExtractor import DataExtractor


def main():
    de = DataExtractor()
    de.DEFAULT_TOP_TOPICS = 40
    de.process_pipeline()
    de.print_topic_distribution()


main()
