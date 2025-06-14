from medical_facilities_list_agent import create_medical_facilities_list_agent
import csv

def csv_to_dict(filename):
    result = {}
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            result[i] = (row['conv_id'], row['cleaned_conversation'])
    return result

def dict_to_csv(data_dict):
    with open("BOT Tools Metrics - 12th June testing RAW - medical_facilities_list agent results.csv", mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['conv_id', 'cleaned_conversation', 'transfer_chat_agent_output'])
        
        # Write rows from dict values (tuples of size 3)
        for key in sorted(data_dict.keys()):
            conv_id, cleaned_conversation, transfer_chat_agent_output = data_dict[key]
            writer.writerow([conv_id, cleaned_conversation, transfer_chat_agent_output])


if __name__ == "__main__":
    chats = csv_to_dict("BOT Tools Metrics - 12th June testing RAW.csv")
    agent = create_medical_facilities_list_agent()
    # outputs = {}
    # for i in range(70):
    #     print(i)
    #     output = agent.invoke({'input': chats[i][1]})
    #     outputs[i] = (chats[i][0], chats[i][1], output['output'])
    #     time.sleep(3)
    # dict_to_csv(outputs)

    print(agent.invoke({'input': """{
  "chat_id": "CH0dc4f033e1294f88936f2a5d37b4b2a9",
  "participants": [
    "Bot",
    "Consumer",
    "System"
  ],
  "conversation": [
    {
      "timestamp": "2025-06-11T23:52:56",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Please i need to see the doctor :tired_face:"
    },
    {
      "timestamp": "2025-06-11T23:55:20",
      "sender": "Bot",
      "type": "normal message",
      "content": "Here are the nearest covered Clinics to you:\r\n(1) Ajman Medical Center - Ajman\r\nAddress: https://www.google.com/maps?q=25.313055,55.5287426\r\nTelephone: 97167430088\r\n\r\n\r\n(2) Al Shamil Medical Centre LLC - Sharjah\r\nAddress: https://www.google.com/maps?q=25.3328059,55.4560571\r\nTelephone: 97165663343\r\n\r\n\r\n(3) Al Nahrain Specialty Medical Center (Al Nuaimia 2) - Ajman\r\nAddress: https://www.google.com/maps?q=25.386551,55.455428\r\nTelephone: 97167449144"
    },
    {
      "timestamp": "2025-06-11T23:55:26",
      "sender": "Bot",
      "type": "normal message",
      "content": "Hello, I am Doctor Mia. What symptoms are you experiencing?"
    },
    {
      "timestamp": "2025-06-12T00:00:24",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Doctor mia mam i already told you about symptoms \r\nThe cutting itching i have in my private part"
    },
    {
      "timestamp": "2025-06-12T00:00:58",
      "sender": "Bot",
      "type": "normal message",
      "content": "When did the cutting pain and itching start?"
    },
    {
      "timestamp": "2025-06-12T00:01:25",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Three weeks ago"
    },
    {
      "timestamp": "2025-06-12T00:01:56",
      "sender": "Bot",
      "type": "normal message",
      "content": "I have already provided you with the nearest clinics. Do you need any more help with the clinic information?"
    },
    {
      "timestamp": "2025-06-12T00:03:21",
      "sender": "Consumer",
      "type": "normal message",
      "content": "No thank alot i was just replying to you're messages you were asking"
    },
    {
      "timestamp": "2025-06-12T00:03:45",
      "sender": "Bot",
      "type": "normal message",
      "content": "I hope I was able to assist you today. If you ever need help or feel unwell, please let me know right away. ItÃ¢Â€Â™s important to speak up so we can guide you and make sure you have all the information and support you need. IÃ¢Â€Â™m always here to help! :blush:"
    },
    {
      "timestamp": "2025-06-12T00:56:08",
      "sender": "Bot",
      "type": "normal message",
      "content": "I hope I was able to assist you today. If you ever need help or feel unwell, please let me know right away. ItÃ¢Â€Â™s important to speak up so we can guide you and make sure you have all the information and support you need. IÃ¢Â€Â™m always here to help! :blush:\r\n\r\n:muscle: Stay Safe While You Work\r\nProtect your body when working. Bend your knees (not your back) when lifting heavy things. Use both hands to carry items. Stand up straight to avoid pain or injury."
    }
  ]
}"""})['output'])