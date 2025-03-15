from flask import Flask, render_template, request
import random
import numpy as np
from database import questions, characters

app = Flask(__name__)

questions_so_far = []
answers_so_far = []
remaining_characters = characters.copy()

@app.route('/')
def index():
    global questions_so_far, answers_so_far, remaining_characters

    question = request.args.get('question')
    answer = request.args.get('answer')
    if question and answer:
        questions_so_far.append(int(question))
        answers_so_far.append(float(answer))

    probabilities = calculate_probabilities(questions_so_far, answers_so_far)
    print("probabilities", probabilities)

    if len(questions_so_far) > 15:
        remaining_characters = [char for char in probabilities if char['probability'] >= 0.05]

    if not remaining_characters:
        return render_template('index.html', result="No matching character found! Try again.")
    
    if len(questions_so_far) >= 20 or any(p.get('probability', 0) > 0.95 for p in remaining_characters):
        result = max(remaining_characters, key=lambda p: p['probability'])
        return render_template('index.html', result=result['name'], questions_so_far=questions_so_far, answers_so_far=answers_so_far)

    next_question = find_best_question()
    if not next_question:
        result = max(remaining_characters, key=lambda p: p['probability'])
        return render_template('index.html', result=result['name'], questions_so_far=questions_so_far, answers_so_far=answers_so_far)

    return render_template('index.html', question=next_question, question_text=questions[next_question])


def calculate_probabilities(questions_so_far, answers_so_far):
    probabilities = []
    for character in remaining_characters:
        probabilities.append({
            'name': character['name'],
            'probability': calculate_character_probability(character, questions_so_far, answers_so_far)
        })
    return probabilities


def calculate_character_probability(character, questions_so_far, answers_so_far):
    P_character = 1 / max(len(remaining_characters), 1)
    log_P_answers_given_character = 0
    log_P_answers_given_not_character = 0

    for question, answer in zip(questions_so_far, answers_so_far):
        certainty = 1 - abs(0.5 - answer)

        log_P_answers_given_character += np.log(max(0.1, 1 - certainty * abs(answer - character_answer(character, question))))

        if len(remaining_characters) > 1:
            P_answer_not_character = np.sum([
                (1 - abs(answer - character_answer(not_character, question))) * (1 / len(remaining_characters))
                for not_character in remaining_characters if not_character['name'] != character['name']
            ])
            log_P_answers_given_not_character += np.log(max(0.1, P_answer_not_character))

    P_answers_given_character = np.exp(log_P_answers_given_character)
    P_answers_given_not_character = np.exp(log_P_answers_given_not_character)

    P_answers = P_character * P_answers_given_character + (1 - P_character) * P_answers_given_not_character
    P_character_given_answers = (P_answers_given_character * P_character) / max(P_answers, 1e-9)  # Prevent div by zero
    return P_character_given_answers


def character_answer(character, question):
    return character.get('answers', {}).get(question, 0.5)


def find_best_question():
    if not remaining_characters:
        return None

    unused_questions = list(set(questions.keys()) - set(questions_so_far))
    if not unused_questions:
        return None

    best_questions = []
    best_score = -1

    for question in unused_questions:
        split_value = calculate_question_split(question)
        expected_answer = np.mean([character_answer(char, question) for char in remaining_characters])

        weighted_split = (1 - abs(0.5 - expected_answer)) * split_value

        if weighted_split > best_score:
            best_score = weighted_split
            best_questions = [question]
        elif weighted_split == best_score:
            best_questions.append(question)

    # Randomly pick among equally good questions
    return random.choice(best_questions) if best_questions else None


def calculate_question_split(question):
    distribution = {0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1: 0}
    for character in remaining_characters:
        distribution[character_answer(character, question)] += 1

    total = len(remaining_characters)
    split_value = sum((count / max(1e-9, total)) ** 2 for count in distribution.values())
    return 1 - split_value


if __name__ == '__main__':
    app.run()