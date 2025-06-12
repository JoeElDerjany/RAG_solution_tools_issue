from transfer_chat_agent import create_transfer_chat_agent
import csv

def csv_to_dict(filename):
    result = {}
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            result[i] = (row['conv_id'], row['cleaned_conversation'])
    return result

def dict_to_csv(data_dict):
    with open("Tool Metrics - June 8 - kartavya's results 2 - transfer_chat agent results.csv", mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['conv_id', 'cleaned_conversation', 'transfer_chat_agent_output'])
        
        # Write rows from dict values (tuples of size 3)
        for key in sorted(data_dict.keys()):
            conv_id, cleaned_conversation, transfer_chat_agent_output = data_dict[key]
            writer.writerow([conv_id, cleaned_conversation, transfer_chat_agent_output])


if __name__ == "__main__":
    chats = csv_to_dict("Tool Metrics - June 8 - kartavya's results 2.csv")
    agent = create_transfer_chat_agent()
    outputs = {}
    for i in range(34):
        print(i)
        output = agent.invoke({'input': chats[i][1]})
        outputs[i] = (chats[i][0], chats[i][1], output['output'])
    dict_to_csv(outputs)