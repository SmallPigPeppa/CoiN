import json, os, argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default="./results/CLIT/ScienceQA/GQA")
    parser.add_argument('--questions', type=str, default='./playground/Instructions/ScienceQA/test.json')
    parser.add_argument('-r', '--rule', default='./llava/eval/table/rule.json')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    answers = open(os.path.expanduser(args.results))
    
    with open(os.path.expanduser(args.questions), "r") as f:
        questions = json.load(f)
    question_dict = {question['question_id']:question for question in questions}

    promts_answers = []

    for i, ans_js in enumerate(answers):
        ans = json.loads(ans_js)
        question_id = ans['question_id']
        question = question_dict[question_id]

        question_label = question['text'].split('\n')[:-1]
        question_label = '\n'.join(question_label)
        answer = ans['text']
        groundtruth = question['answer']

        system_dict = {"role": "system",
                    "content": "You are a helpful and precise assistant for checking the quality of the answer.",}
        
        rule = rule_dict['CLIT']
        prompt = rule['prompt']

        content = (f'[Context]\n'
                   f'[Question]\n{question_label}\n\n'
                   f'[Ground truth answer]\n{groundtruth}\n\n[End of ground truth answer]\n\n'
                   f'[{rule["role"]}]\n{answer}\n\n[End of {rule["role"]}]\n\n'
                   f'[System]\n{prompt}\n\n')
        
        user_dict = { "role": "user", "content": content}
        message = [system_dict, user_dict]
        promts_answers.append(message)

    path = os.path.split(args.results)[0]
    json.dump(promts_answers,open(os.path.join(path,'prompt_to_eval.json'),'w'),indent=4)