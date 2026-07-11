"""
    Prompt is divided into two parts:
    1. INSTRUCTIONS - which remain the same for every request sent to LLM
    2. USER QUERY - which changes with every request
"""

INSTRUCTIONS = """
                    Your task is to answer questions from the course participants
                    based on the provided context.

                    Use the context to find relevant information and provide accurate
                    answers. If the answer is not found in the context,
                    respond with "I don't know."
                """

USER_QUERY_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


class RAGBase:
    '''
        An encapsulation of all things needed by a basic rag
    '''

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_QUERY_TEMPLATE,
        course="llm-zoomcamp",
        model="llama3.1",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 2.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )


    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()


    def build_prompt(self, question, search_results):
        context = self.build_context(search_results=search_results)
        return self.prompt_template.format(question=question, context=context)
    
    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text


    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

