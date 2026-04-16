import numpy as np


letters = {
    "О": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1]
    ]),
    
    "Л": np.array([
        [-1, -1, -1,  1,  1],
        [-1, -1,  1, -1,  1],
        [-1,  1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1]
    ]),
    
    "Е": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1]
    ]),
    
    "Г": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1]
    ])
}

# 2. Вихідні вектори (біполярне кодування)
# 4 вихідні нейрони

targets = {
    "О": np.array([ 1, -1, -1, -1]),
    "Л": np.array([-1,  1, -1, -1]),
    "Е": np.array([-1, -1,  1, -1]),
    "Г": np.array([-1, -1, -1,  1])
}


def flatten_letter(letter_matrix):
    """Перетворює 5x5 у вектор з 25 елементів."""
    return letter_matrix.reshape(-1)

def sign_bipolar(x):
    """Біполярна активація."""
    return np.where(x >= 0, 1, -1)

def print_letter(letter_matrix):
    """Красивий друк букви в консолі."""
    for row in letter_matrix:
        print(" ".join("■" if x == 1 else "·" for x in row))
    print()

# 4. Підготовка навчальної вибірки

X = []
Y = []
labels = []

for label in ["О", "Л", "Е", "Г"]:
    X.append(flatten_letter(letters[label]))
    Y.append(targets[label])
    labels.append(label)

X = np.array(X)   # shape = (4, 25)
Y = np.array(Y)   # shape = (4, 4)

# W = sum(x^T * y)

num_inputs = X.shape[1]    # 25
num_outputs = Y.shape[1]   # 4

W = np.zeros((num_inputs, num_outputs))
b = np.zeros(num_outputs)

for x, y in zip(X, Y):
    W += np.outer(x, y)
    b += y


def recognize(input_vector):
    net = np.dot(input_vector, W) + b
    output = sign_bipolar(net)
    return net, output

def decode_output(output_vector):
    for label, target in targets.items():
        if np.array_equal(output_vector, target):
            return label
    return "Невідомий символ"

#  Перевірка

print("=== НАВЧАЛЬНІ СИМВОЛИ ===\n")

for label in labels:
    print(f"Символ: {label}")
    print_letter(letters[label])

print("=== РЕЗУЛЬТАТИ РОЗПІЗНАВАННЯ ===\n")

for label in labels:
    x = flatten_letter(letters[label])
    net, out = recognize(x)
    predicted = decode_output(out)

    print(f"Вхідний символ: {label}")
    print(f"Вихід мережі (net): {net}")
    print(f"Біполярний вихід: {out}")
    print(f"Розпізнано як: {predicted}")
    print("-" * 50)

test_O = letters["О"].copy()
test_O[2, 2] = 1   # трохи змінили центр

print("\n=== ТЕСТ НА СПОТВОРЕНОМУ СИМВОЛІ ===\n")
print("Спотворена буква О:")
print_letter(test_O)

net, out = recognize(flatten_letter(test_O))
predicted = decode_output(out)

print(f"Вихід мережі (net): {net}")
print(f"Біполярний вихід: {out}")
print(f"Розпізнано як: {predicted}")

# =========================================
# 9. Нерозв’язна ситуація адаптації
# Один і той самий вхід -> різні бажані виходи
# Це неможливо коректно навчити
# =========================================

print("\n=== НЕРOЗВ’ЯЗНА СИТУАЦІЯ АДАПТАЦІЇ ===\n")

conflict_input = flatten_letter(letters["О"])
desired_output_1 = targets["О"]
desired_output_2 = targets["Л"]

print("Один і той самий вхідний символ подається з двома різними правильними відповідями:")
print("Вхід = буква О")
print(f"Бажаний вихід 1: {desired_output_1}  -> клас О")
print(f"Бажаний вихід 2: {desired_output_2}  -> клас Л")

if np.array_equal(conflict_input, flatten_letter(letters["О"])) and not np.array_equal(desired_output_1, desired_output_2):
    print("\nВисновок:")
    print("Для одного й того самого вхідного вектора задано різні цільові виходи.")
    print("Така задача є нерозв’язною для мережі Хебба, бо однаковий вхід не може стабільно відповідати двом різним класам.")