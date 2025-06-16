from GraphRAG_func import createAgent
import csv

def csv_to_dict(filename):
    result = {}
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            result[i] = (row['conv_id'], row['cleaned_conversation'])
    return result

def dict_to_csv(data_dict):
    with open("BOT Tools Metrics - 12th June testing RAW - all tools agent results.csv", mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['conv_id', 'cleaned_conversation', 'agent_output'])
        
        # Write rows from dict values (tuples of size 3)
        for key in sorted(data_dict.keys()):
            conv_id, cleaned_conversation, transfer_chat_agent_output = data_dict[key]
            writer.writerow([conv_id, cleaned_conversation, transfer_chat_agent_output])


if __name__ == "__main__":
    chats = csv_to_dict("BOT Tools Metrics - 12th June testing RAW.csv")
    agent = createAgent()
    # outputs = {}
    # for i in range(70):
    #     print(i)
    #     output = agent.invoke({'input': chats[i][1]})
    #     outputs[i] = (chats[i][0], chats[i][1], output['output'])
    # dict_to_csv(outputs)

    print(agent.invoke({'input': """{
  "chat_id": "CH14d2dadc1535456dac866734508d51c2",
  "participants": [
    "Agent",
    "Consumer",
    "System"
  ],
  "conversation": [
    {
      "timestamp": "2025-06-11T17:54:24",
      "sender": "Agent",
      "type": "normal message",
      "content": "In case of emergencies, which are life-threatening cases like accidents, severe bleeding, heart attacks, or blood pressure higher than 200/100..you can take her to any hospital. and they will follow the emergency protocol \r\nBut it is always better to take her to a covered hospital \r\nThis is the hospital's link to know what hospitals are under the insurance network :\r\n:hotel: Hospitals: https://maids.page.link/rtBLkbr8YXN5V1yi8"
    },
    {
      "timestamp": "2025-06-11T17:54:44",
      "sender": "Consumer",
      "type": "normal message",
      "content": "Ok"
    },
    {
      "timestamp": "2025-06-11T17:55:18",
      "sender": "Agent",
      "type": "normal message",
      "content": "In any other case, your nanny should always visit the clinic first and have a check-up with the general doctor first. \r\nare you asking in general please?  or is your nanny sick?"
    },
    {
      "timestamp": "2025-06-11T18:29:28",
      "sender": "Agent",
      "type": "normal message",
      "content": "I hope I was able to assist you today. If you have any further concerns or your maid ever feels unwell, donÃ¢Â€Â™t hesitate to reach out. I'm always here to help!:blush:"
    }
  ]
}"""})['output'])