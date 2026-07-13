import requests
import time

from sqlitesearch import TextSearchIndex

'''
The faqs data is cleaned and prepared by course owner, and provided to us,
by the course owner of datatalk.clubs. So, that we can work on LLM related
stuff. But, most of our work goes in preparing the data and get it indexed
with right database for our purpose.
'''

def load_faq_data():
    courses_list_url = "https://datatalks.club/faq/json/courses.json"
    courses_list_response = requests.get(courses_list_url)
    courses_list_response.raise_for_status()
    courses_list = courses_list_response.json()


    faqs_data = []
    faq_url_prefix = "https://datatalks.club/faq"

    for course in courses_list:

        course_info_response = requests.get(f"{faq_url_prefix}{course['path']}")
        course_info_response.raise_for_status()
        course_faqs = course_info_response.json()

        faqs_data.extend(course_faqs)
    
    return faqs_data


def index_data_to_sql():
    faqs_data = load_faq_data()
    print(f"Loaded {len(faqs_data)} documents")


    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path="faq.db"
    )

    for doc in faqs_data:
        index.add(doc)
        print(f"""Added: {doc["question"][:60]}...""")
        time.sleep(0.5)

    index.close()
    print("Done. Index saved to faq.db")

