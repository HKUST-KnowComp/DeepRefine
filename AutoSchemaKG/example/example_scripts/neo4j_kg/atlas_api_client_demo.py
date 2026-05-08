from openai import OpenAI
message = [
    {
        'role':'system',
        'content':'The following are multiple choice questions (with answers) about miscellaneous.'
    },
    {
        'role':'user', 
        'content':"""Given the following question and four candidate answers (A, B, C and D), choose the answer.
Question: The openSSL implementation described in “Remote Timing Attacks are Practical” (by Brumley and Boneh) uses the following performance optimizations: Chinese Remainder (CR), Montgomery Representation (MR), Karatsuba Multiplication (KM), and Repeated squaring and Sliding windows (RS). Which of the following options would close the timing channel attack described in the paper if you turned the listed optimizations off?
1. RS and KM
2. RS and MR
A. True, True
B. False, False
C. True, False
D. False, True"""
    }
]

wiki_base_url = "http://0.0.0.0:10087/v1/"
wiki_client = OpenAI(api_key="EMPTY", base_url=wiki_base_url)
pes2o_base_url = "http://0.0.0.0:10088/v1/"
pes2o_client = OpenAI(api_key="EMPTY", base_url=pes2o_base_url)
cc_base_url ="http://0.0.0.0:10089/v1/"
cc_client = OpenAI(api_key="EMPTY", base_url=cc_base_url)

# knowledge graph en_simple_wiki_v0
for client in [wiki_client, pes2o_client, cc_client]:
    response = client.chat.completions.create(
        model="llama",
        messages=message,
        max_tokens=2048,
        temperature=0.5,
    )
    print(response.choices[0].message.content)