import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field
from typing import List
from langchain_community.vectorstores import Neo4jVector

def create_send_document_agent():
 # retrieve Neo4j credentials and configuration from environment variables
    NEO4J_URI = os.environ["NEO4J_URI"]
    NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

    # retrieve OpenAI API key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # initialize OpenAI embeddings and the GPT-4 model
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
    # connect to the Neo4j graph database
    kg = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    # VERY IMPORTANT: If the Neo4j database instance is idle for a while, it'll be paused. If this code is run while the instance is paused,
    # it'll throw an error. In case you encounter an error related to the Neo4j database, contact the person with access to the database
    # so that he/she can resume running the instance. As of now, that person is Joe El-Derjany.

    # initialize a Neo4j vector index for hybrid search
    textbook_graphrag_vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )

    # define a Pydantic model for entity extraction
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description=
            """Every action, amount, api, application, argument, attribute, body part, brand, business_function, button, concept, concern, condition, contact,
            day, decision, diagnosis, document, end, entity, event, facility, financial product, identifier, instruction, insurance, language, link,
            location, medical treatment, medication, message, message variant, method, name, object, occupation, organization, parameter, patient,
            percentage, person, place, policy, process, product, prompt, protocol, region, requirement, resource, rule, service, specialty, string,
            substance, symptom, system, team, technical_function, time, tool, treatment, trigger, unknown, user relationship, and value that appears in the conversation."""
        )

    # create a chain to extract structured entity information with the LLM
    entity_chain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Every action, amount, api, application, argument, attribute, body part, brand, business_function, button, concept, concern, condition, contact,
            day, decision, diagnosis, document, end, entity, event, facility, financial product, identifier, instruction, insurance, language, link,
            location, medical treatment, medication, message, message variant, method, name, object, occupation, organization, parameter, patient,
            percentage, person, place, policy, process, product, prompt, protocol, region, requirement, resource, rule, service, specialty, string,
            substance, symptom, system, team, technical_function, time, tool, treatment, trigger, unknown, user relationship, and value that appears in the conversation.""",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = entity_chain_prompt | llm.with_structured_output(Entities)

    # Create a full-text index named 'entity' on the 'id' property of nodes labeled '__Entity__'
    # This index enables efficient text-based searches on the 'id' property.
    # The 'IF NOT EXISTS' clause ensures the index is only created if it doesn't already exist.
    kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # function to generate a full-text search query for Neo4j, to be used in the structured retriever function below
    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # function to retrieve structured data from Neo4j based on entities
    def structured_retriever(question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            print(f" Getting Entity: {entity}")
            response = kg.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el["output"] for el in response])
        return result

    # function to retrieve unstructured data from Neo4j
    def unstructured_retriever(question: str) -> str:
        unstructured_data = [
            el.page_content for el in textbook_graphrag_vector_index.similarity_search(question)
        ]
        return "".join(word.replace("\n", " ") for word in unstructured_data)

    # create a custom GraphRAG prompt based on the predefined ReAct prompt 
    textbook_graphrag_template = """
    <prompt>
    # TOOL AUDITOR PROMPT

    <role>
    ## ROLE
    You are 'SendDocumentToolAuditBot', an assistant specialized in auditing LLM conversations to verify correct usage of the 'send_document' tool. 
    'send_document' is a tool discussed and explained in a Neo4j database.
    </role>

    <questions>
    ## QUESTIONS
    1. Does the conversation include any situation, described in the Neo4j database, that requires the 'send_document' tool to be called? 
    2. How many times SHOULD it have been called?
    </questions>

    <task>
    ## TASK
    Always adhere to the following, in order, to verify the tool usage. Conside the CRITICAL NOTES below in the '<task>' section for every step:
    1. Answer the questions in '<questions>' by:
        a. Calling the 'Structured_GraphRAG' tool to search the database through entity-relationship traversal for all relevant data regarding the 'send_document' tool.
        b. Then, Calling the 'Unstructured_GraphRAG' tool to search the database through similarity-search for all relevant data regarding the 'send_document' tool.
        c. Finally, combinig the outputs of both tools to formulate an answer to every question.
    2. Then, output your final answer as a single JSON object (no extra text), matching exactly the 'Output Schema' below.

    CRITICAL NOTES: 
    1. For 'numberTimes_Supposed_To_Be_Called': If a request or trigger appears multiple times (even if repeated in adjacent messages), consider each as a SEPARATE and independent reason to call the tool AND increase the count for 'numberTimes_Supposed_To_Be_Called'
    2. The 'send_document' tool is ONLY called when a CONVERSATION is with the BOT, (that is only for BOT and CONSUMER conversation). IF a conversation is being handled by an AGENT, THE TOOL IS NEVER SUPPOSED TO BE CALLED, even if tools conditions are met.
    3. When making 'SHOULD have been called' decisions, think step-by-step:
        - Identify context cues or user requests requiring a tool, keeping the '<guiding_principle>' in mind.
        - Think like a human reviewer, infer intent from broken grammar, multiple languages, or indirect phrases.
    </task>

    <input_details>
    ## INPUT
    Input will be a conversation log (JSON) between a consumer and a representative of maids.cc (Agent, Bot, or System). The 'conversation' array contains entries with fields: timestamp, sender, type (private or normal or transfer msg or tool), tool (if and only if, type=='tool'), and content. Use only the entries and fields as per rules to audit tool usage.
    </input_details>

    <expected_output>
    ## OUTPUT SCHEMA
    The output must follow this JSON structure and key order:

    [
    {
        'chatId': 'string (e.g., b29LvU21PaC97235N9)',

        'send_document': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        },
    }
    ]

    </expected_output>

    <your_tools>
    ## YOUR TOOLS
    - 'Structured_GraphRAG': Retrieves all relevant data regarding the 'send_document' tool from the database through entity-relationship traversal.
    - 'Unstructured_GraphRAG': Retrieves all relevant data regarding the 'send_document' tool from the database through similarity search.
    </tools>

    </prompt>

    """

    react_prompt = hub.pull("langchain-ai/react-agent-template")
    custom_textbook_graphrag_prompt = react_prompt.partial(instructions=textbook_graphrag_template)

    # create tools for structured and unstructured retrieval
    structured_retrieval_tool = Tool(
        name="Structured_GraphRAG",
        func=structured_retriever,
        description="Retrieves all relevant data regarding the 'send_document' tool from the database through entity-relationship traversal.",
    )
    unstructured_retrieval_tool = Tool(
        name="Unstructured_GraphRAG",
        func=unstructured_retriever,
        description="Retrieves all relevant data regarding the 'send_document' tool from the database through similarity search.",
    )

    # create a GraphRAG agent using the tools, LLM, and custom prompt
    textbook_graphrag_agent = create_react_agent(
            tools=[structured_retrieval_tool, unstructured_retrieval_tool], llm=llm, prompt=custom_textbook_graphrag_prompt
        )

    # create an executor to run the GraphRAG agent
    textbook_graphrag_agent_executor = AgentExecutor(
        agent=textbook_graphrag_agent,
        tools=[structured_retrieval_tool, unstructured_retrieval_tool],
        verbose=True,
        handle_parsing_errors=True,
        output_parser = StrOutputParser()
    )

    return textbook_graphrag_agent_executor