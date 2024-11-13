from SimpleLinearRegression import *
import sys

def main():
    # x_label, x_train_set = 'advertising', (23, 26, 30, 34, 43, 48, 52, 57, 58)
    # y_label, y_train_set = 'sales', (651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518)

    x_label, x_train_set = 'advertising', (1, 2, 3, 4, 5, 6, 7, 8, 9)
    y_label, y_train_set = 'sales', (2, 4, 6, 8, 10, 12, 14, 16, 18)

    prediction_set = map(int, sys.argv[1:])

    SLR = SimpleLinearRegression(x_train_set, y_train_set)

    model = SLR.train()

    print('Hands-On 2 - Simple Linear Regression\n')

    print('La ecuación de regresión para el conjunto de datos es:\n')
    print(f'{y_label} = {x_label} * {model[1]} + {model[0]}\n')    

    for prediction in prediction_set:
        result = f'La predicción para el valor {prediction} de {x_label} es: {SLR.predict(prediction)}'

        if prediction in x_train_set:
            y_value = y_train_set[x_train_set.index(prediction)]
            print(f'{result} y se tiene como valor real: {y_value}')
        else:
            print(result)


if __name__ == "__main__":
    main()