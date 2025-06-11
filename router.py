import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from transfer_chat_agent import create_transfer_chat_agent
from send_document_agent import create_send_document_agent
from medical_facilities_list_agent import create_medical_facilities_list_agent
from open_a_complaint_agent import create_open_a_complaint_agent
from insurance_covered_agent import create_insurance_covered_agent

def createAgent():
    transfer_chat_agent = create_transfer_chat_agent()
    send_document_agent = create_send_document_agent()
    medical_facilities_list_agent = create_medical_facilities_list_agent()
    open_a_complaint_agent = create_open_a_complaint_agent()
    insurance_covered_agent = create_insurance_covered_agent()

    def tc_func(chat:str) -> str:
        return transfer_chat_agent.invoke({"input": chat})["output"]
    def sd_func(chat:str) -> str:
        return send_document_agent.invoke({"input": chat})["output"]
    def mfl_func(chat:str) -> str:
        return medical_facilities_list_agent.invoke({"input": chat})["output"]
    def oc_func(chat:str) -> str:
        return open_a_complaint_agent.invoke({"input": chat})["output"]
    def ic_func(chat:str) -> str:
        return insurance_covered_agent.invoke({"input": chat})["output"]

    # retrieve OpenAI API key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")

    # create a custom prompt based on the predefined ReAct prompt 
    template = """
    <prompt>
    # TOOL AUDITOR PROMPT

    <role>
    ## ROLE
    You are 'ToolAuditBot', an assistant specialized in auditing LLM conversations to verify correct tool usage, for a company named maids.cc. 
    </role>

    <guiding_principle>
    ## GUIDING PRINCIPLE: BOT-ONLY TOOL USAGE
    This is the most important rule. Before evaluating any tool triggers, you must first ensure that conversation is with the tool by checking the “sender” = “BOT”. If the conversation is transferred or handled by anyone else EXCEPT BOT (that is where “sender” = “AGENT”), a TOOL should NEVER be called, even if the conditions described below are met. The human agent follows different procedures.

    **A tool is ONLY supposed to be called when the conversation is with the BOT.**
    </guiding_principle>

    <task>
    ## TASK
    Always adhere to the following, in order, to verify the tool usage. Conside the CRITICAL NOTES below in the '<task>' section for every step:
    1. First, apply the '<guiding_principle>' to the entire conversation and then carefully read the entire conversation JSON provided by the user.
    2. Call the Transfer_Chat_Audit_Tool to verify correct usage of the 'transfer_chat' tool.
    3. Call the Send_Document_Audit_Tool to verify correct usage of the 'send_document' tool.
    4. Call the Medical_Facilities_List_Audit_Tool to verify correct usage of the 'medical_facilities_list' tool.
    5. Call the Open_a_Complaint_Audit_Tool to verify correct usage of the 'open_a_complaint' tool.
    6. Call the Insurance_Covered_Audit_Tool to verify correct usage of the 'insurance_covered' tool.
    7. Use the outputs of all the tools to formulate your final output. Your final answer MUST be a single JSON object (no extra text) matching exactly the 'Output Schema' below.
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

        'transfer_chat': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        },

        'send_document': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        },

        'medical_facilities_list': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        },

        'open_a_complaint': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        },

        'insurance_covered': {
        'Supposed_To_Be_Called': boolean (true/false),
        'numberTimes_Supposed_To_Be_Called': integer,
        'reason_Supposed_To_Be_Called': string
        }
    }
    ]

    </expected_output>

    <tools>
    ## TOOLS
    - 'Transfer_Chat_Audit_Tool': Verifies correct usage of the 'transfer_chat' tool.
    - 'Send_Document_Audit_Tool': Verifies correct usage of the 'send_document' tool.
    - 'Medical_Facilities_List_Audit_Tool': Verifies correct usage of the 'medical_facilities_list' tool.
    - 'Open_a_Complaint_Audit_Tool': Verifies correct usage of the 'open_a_complaint' tool.
    - 'Insurance_Covered_Audit_Tool': Verifies correct usage of the 'insurance_covered' tool.
    </tools>

    </prompt>

    """

    react_prompt = hub.pull("langchain-ai/react-agent-template")
    custom_prompt = react_prompt.partial(instructions=template)

    # create tools for structured and unstructured retrieval
    transfer_chat_audit_tool = Tool(
        name="Transfer_Chat_Audit_Tool",
        func=tc_func,
        description="Verifies correct usage of the 'transfer_chat' tool. Takes the whole input chat as input.",
    )
    send_document_audit_tool = Tool(
        name="Send_Document_Audit_Tool",
        func=sd_func,
        description="Verifies correct usage of the 'send_document' tool. Takes the whole input chat as input.",
    )
    medical_facilities_list__audit_tool = Tool(
        name="Medical_Facilities_List_Audit_Tool",
        func=mfl_func,
        description="Verifies correct usage of the 'medical_facilities_list' tool. Takes the whole input chat as input.",
    )
    open_a_complaint_audit_tool = Tool(
        name="Open_a_Complaint_Audit_Tool",
        func=oc_func,
        description="Verifies correct usage of the 'open_a_complaint' tool. Takes the whole input chat as input.",
    )
    insurance_covered_audit_tool = Tool(
        name="Insurance_Covered_Audit_Tool",
        func=ic_func,
        description="Verifies correct usage of the 'insurance_covered' tool. Takes the whole input chat as input.",
    )

    # create a GraphRAG agent using the tools, LLM, and custom prompt
    agent = create_react_agent(
            tools=[transfer_chat_audit_tool, send_document_audit_tool, medical_facilities_list__audit_tool, open_a_complaint_audit_tool, insurance_covered_audit_tool], llm=llm, prompt=custom_prompt
        )

    # create an executor to run the GraphRAG agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[transfer_chat_audit_tool, send_document_audit_tool, medical_facilities_list__audit_tool, open_a_complaint_audit_tool, insurance_covered_audit_tool],
        verbose=True,
        handle_parsing_errors=True,
        output_parser = StrOutputParser()
    )

    return agent_executor
