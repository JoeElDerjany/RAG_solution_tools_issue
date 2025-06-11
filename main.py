from GraphRAG_func import createAgent

if __name__ == "__main__":
    agent = createAgent()
    output = agent.invoke({"input": """{
  "chat_id": "CH11b356dec1a74deca3cbc7db426fd655",
  "participants": [
    "Agent",
    "Consumer",
    "System"
  ],
  "conversation": [
    {
      "timestamp": "2025-06-08T18:03:22",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Good day! May I know if my Insurance card covered the pap smear procedure here in Houston Clinic Albarsha?"
    },
    {
      "timestamp": "2025-06-08T18:03:39",
      "sender": "System",
      "type": "private message",
      "content": "applicant -"
    },
    {
      "timestamp": "2025-06-08T18:03:51",
      "sender": "System",
      "type": "transfer",
      "content": "Conversation transferred from skill DOCTORS_BOT to skill GPT_Doctors By TWILIO_FLOW Displaying the chat to Ã¢Â€ÂœAmal.AbÃ¢Â€Â under the skill 'Doctor'."
    },
    {
      "timestamp": "2025-06-08T18:04:15",
      "sender": "System",
      "type": "transfer",
      "content": "Conversation transferred from skill GPT_Doctors to skill Doctor By ERP_chatai and selecting the user: Amal.Ab"
    },
    {
      "timestamp": "2025-06-08T18:08:23",
      "sender": "Agent",
      "type": "normal message",
      "content": "Good day to you too. This is Dr. Mia, May I know What are you feeling?"
    },
    {
      "timestamp": "2025-06-08T18:11:42",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Im good, but could you please help me to confirm if the pap smear procedure is covered by my insurance?"
    },
    {
      "timestamp": "2025-06-08T18:13:49",
      "sender": "Agent",
      "type": "normal message",
      "content": "Routine examination and check up are not covered by insurance."
    },
    {
      "timestamp": "2025-06-08T18:15:00",
      "sender": "Consumer",
      "type": "normal message",
      "content": "ok thank you"
    },
    {
      "timestamp": "2025-06-08T18:53:38",
      "sender": "Agent",
      "type": "normal message",
      "content": "I hope I was able to assist you today. If you ever need help or feel unwell, please let me know right away. ItÃ¢Â€Â™s important to speak up so we can guide you and make sure you have all the information and support you need. IÃ¢Â€Â™m always here to help! :blush:\n\nStay Safe in the Kitchen!\nYour hands are your greatest toolsÃ¢Â€Â”protect them! When using a knife, stay focused and avoid distractions to prevent accidents. Your safety is always my top priority."
    },
    {
      "timestamp": "2025-06-08T18:53:46",
      "sender": "System",
      "type": "transfer",
      "content": "Conversation transferred from skill Doctor to skill Doctor By Amal.Ab and selecting the user: MohammadBa"
    }
  ]
}"""})
    print(agent.agent.llm_chain.prompt.template)
   # print(output)