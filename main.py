from GraphRAG_func import createAgent

if __name__ == "__main__":
    agent = createAgent()
    output = agent.invoke({"input": """{
  "chat_id": "CH03c1b42beb7d4b32a11a6aa1f89016a4",
  "participants": [
    "Bot",
    "Consumer",
    "System"
  ],
  "conversation": [
    {
      "timestamp": "2025-06-08T20:03:42",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Good evening  ma'am/sir:hearts:Ã¯Â¸Â\nI just want to consult I have pain  my gums is  swalen is painful ,is any appointments for check up.is this covered for my insurance if go to   doctor to check-up?"
    },
    {
      "timestamp": "2025-06-08T20:03:59",
      "sender": "System",
      "type": "private message",
      "content": "MV Active Maid - WITH_CLIENT"
    },
    {
      "timestamp": "2025-06-08T20:04:01",
      "sender": "Consumer",
      "type": "normal message",
      "content": "This rodelia tamayo"
    },
    {
      "timestamp": "2025-06-08T20:04:11",
      "sender": "System",
      "type": "transfer",
      "content": "Conversation transferred from skill DOCTORS_BOT to skill GPT_Doctors By TWILIO_FLOW Displaying the chat to Ã¢Â€ÂœMohammadBaÃ¢Â€Â under the skill 'Doctor'."
    },
    {
      "timestamp": "2025-06-08T20:04:58",
      "sender": "Bot",
      "type": "normal message",
      "content": "Hello, I am Doctor Mia.\n\nThe insurance will cover 80% of clinic visits, and 70% of the cost if you get medicine from the clinic with a prescription. The insurance is under NAS managed by Orient.\n\nFor a toothache, it is best to see a dentist. We can help you find the nearest dental clinic. Can you please share your full address, including the street, building, and city, so I can find the closest dental clinics for you?"
    },
    {
      "timestamp": "2025-06-08T20:20:48",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Palm Jumeirah \nShoreline 7 Al hasser \nG02"
    },
    {
      "timestamp": "2025-06-08T20:21:34",
      "sender": "Bot",
      "type": "normal message",
      "content": "Here are the nearest covered Dental to you:\n(1) Lotus Medical Center (Al Quoz 3) Dubai\nAddress: https://www.google.com/maps?q=25.157133,55.247372\nTelephone: 97143465380\n\n\n(2) FAITH JABAL MEDICAL CENTER (JABAL ALI) DUBAI\nAddress: https://www.google.com/maps?q=25.010200678956,55.1060588599108\nTelephone: 97143956479\n\n\n(3) Jupiter Speciality Medical Center - Dubai\nAddress: https://www.google.com/maps?q=25.244862,55.298033\nTelephone: 97143866330"
    },
    {
      "timestamp": "2025-06-08T20:21:41",
      "sender": "Bot",
      "type": "normal message",
      "content": "We are happy to help you anytime, feel free to ask us any questions :blush:"
    },
    {
      "timestamp": "2025-06-08T20:22:24",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Thank you :pray:"
    },
    {
      "timestamp": "2025-06-08T20:22:47",
      "sender": "Bot",
      "type": "normal message",
      "content": "You are welcome! :blush:"
    },
    {
      "timestamp": "2025-06-08T21:13:43",
      "sender": "Bot",
      "type": "normal message",
      "content": "I hope I was able to assist you today. If you ever need help or feel unwell, please let me know right away. ItÃ¢Â€Â™s important to speak up so we can guide you and make sure you have all the information and support you need. IÃ¢Â€Â™m always here to help! :blush:\n\n:muscle: Stay Safe While You Work\nProtect your body when working. Bend your knees (not your back) when lifting heavy things. Use both hands to carry items. Stand up straight to avoid pain or injury."
    }
  ]
}"""})
    print(output["output"])