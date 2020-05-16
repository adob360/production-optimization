import numpy as np
import pandas as pd
from sklearn import neighbors, linear_model, svm
from itertools import product


def read_file():
    '''Liest ein Excel File ein und hängt die Tabellenblätter untereinander zusammen.'''
    def is_equal_to_x(value):
        return value == 'x'

    layers = [
        pd.read_excel(
            "Produktionsdaten.xlsx",
            sheet_name=str(sheet_num) + ". Schicht",
            converters={
                'offene Leimfugen': is_equal_to_x,
                'Hit & Miss': is_equal_to_x,
                'seitl. Lamellenversatz': is_equal_to_x,
            },
        )
        for sheet_num in range(1, 6)
    ]
    column_renamings = {'Lamellenbreite - Fertigmaß [mm]': 'Lamellenbreite [mm]'}

    # NOTE: rename works by copy be default, could also have used in_place=True
    transformed_layers = [
        layer.rename(columns=column_renamings) for layer in layers
    ]
    conc_sheet = pd.concat(transformed_layers, axis='index')


    return conc_sheet


def drop_useless_rows(df, *columns):
    '''Erstellt einen Dataframe, der nur aus den gewünschten Spalten besteht.
    In diesem Dataframe werden alle Zeilen, die fehlende Werte enthalten gelöscht.'''
    reduced_df = df[ list(columns) ]
    clean_df = reduced_df.dropna(axis=0)
    return clean_df


def trainer(X, y):
    '''Speichert die Informationen, welches Ergebnis y hat wenn man X einen bestimmten Wert hat.
    Hier könnte man auch statt 'svm.SVR' 'linear_model.LinearRegression' einsetzen'''
    clf = svm.SVR(kernel='linear')
    clf.fit(X, y)
    return clf


def nearest_neighbor(X, y):
    '''Speichert die Information, welcher Klasse y zugeordnet werden kann,
     wenn X einen bestimmten Wert hat.'''
    clf = neighbors.KNeighborsClassifier(n_jobs=-1) #n_jobs sagt, wieviele jobs er gleichzeitig macht
    clf.fit(X, y)
    return clf


def values_in_list(two_d_list):
    '''Nimmt eine zweidimensionale Liste, dreht sie um 90 Grad und schaut,
    dass ein Wert in einer Zeile nur einmal vorkommt.'''
    arr = np.array(two_d_list).T
    value_lists = []
    for values in arr:
        value_list = []
        check_list = []
        for value in values:
            if value not in check_list:
                value_list.append(value)
                check_list.append(value)
        if value_list:
            value_lists.append(value_list)
    return value_lists


def all_combinations(df, *params):
    '''Gibt alle möglichen Kombinationen von einer variablen Anzahl an Listen zurück'''
    value_list = [df[i].unique() for i in params]
    if len(value_list) == 1:
        return "You are an idiot, you should at least enter two parameters!!"
    else:
        prod = list(product(*value_list))
        return prod


def find_best_combo(prod, classifier, minimum=True):
    '''Verwendet den Classifier von der Funktion trainer() und sagt für jede Liste
    einen ein Ergebnis voraus. Dieses wird in einer Liste gespeichert und zum Schluss
    wird je nachdem ob man minimieren oder maximieren will, der minimale oder der maximale
    Wert herausgesucht und die Liste, auf die man diesen Wert prognistizierte zurückgegeben.'''
    predictions = [
        classifier.predict(np.array(i).reshape(1, -1))
        for i in prod
    ]
    if minimum:
        index = np.argmin(predictions)
    else:
        index = np.argmax(predictions)
    return list(prod[index])


def possible_combinations(prod, classifier):
    '''Verwendet den Classifier aus der nearest_neighbor() Funktion
    und gibt eine Liste mit allen Kombinationen zurück, die das gewünschte
    Ergebnis geliefert haben.'''
    predictions = [
        {
            "params": i,
            "value": classifier.predict(np.array(i).reshape(1, -1))
        }
        for i in prod

    ]
    all_combos = [
        prediction['params']
        for prediction in predictions
        if prediction['value'] == False
    ]
    possible_combos = values_in_list(all_combos)
    return possible_combos


def predict_linear(conc_sheet, y_name, x_names, minimum=True):
    '''Funktion, die alle oben definierten Funktionen verwendet.'''
    clean_df = drop_useless_rows(conc_sheet, y_name, *x_names)
    X = np.array([clean_df[i] for i in x_names]).T
    y = np.array(clean_df[y_name])
    clf = trainer(X, y)
    prod = all_combinations(clean_df, *x_names)
    values = find_best_combo(prod, clf, minimum)
    return values


def predict_group(conc_sheet, y_name, x_names):
    '''Funktion, die alle oben definierten Funktionen verwendet.'''
    X = np.array([conc_sheet[i] for i in x_names]).T
    y = np.array(conc_sheet[y_name])
    clf = nearest_neighbor(X, y)
    prod = all_combinations(conc_sheet, *x_names)
    possible_combos = possible_combinations(prod, clf)
    return possible_combos


if __name__ == '__main__':
    # read file only once because it's not fast
    conc_sheet = read_file()

    hitmiss = predict_group(
                      conc_sheet=conc_sheet,
                      y_name='Hit & Miss',
                      x_names=[
                          'Lamellenbreite [mm]',
                          'Hobelmaß Lamellenhobel Breitseite [mm]',
                          'Hobelmaß Keilzinkung Breitseite [mm]',
                          'Hobelmaß Binderhobel Höhe [mm]'
                      ])



    hitmiss = predict_linear(
                      conc_sheet=conc_sheet,
                      y_name='Hit & Miss',
                      x_names=[
                          'Lamellenbreite [mm]',
                          'Hobelmaß Lamellenhobel Breitseite [mm]',
                          'Hobelmaß Keilzinkung Breitseite [mm]',
                          'Hobelmaß Binderhobel Höhe [mm]'
                      ])
    festigkeit = predict_linear(
                         conc_sheet=conc_sheet,
                         y_name="Festigkeit %",
                         x_names=[
                             "Scanparameter Röntgen",
                             "Scanparameter Laser",
                             "Scanparameter Farbkamera",
                         ],
                         minimum=False)

    print(f"Scanparameter Röntgen: {festigkeit[0]} \n"
          f"Scanparameter Laser: {festigkeit[1]} \n"
          f"Scanparameter Farbkamera: {festigkeit[2]} \n \n")


    ausbeute = predict_linear(
                       conc_sheet=conc_sheet,
                       y_name="Ausbeute nach Fehlerkappung [%]",
                       x_names=[
                           "Scanparameter Röntgen",
                           "Scanparameter Laser",
                           "Scanparameter Farbkamera",
                       ],
                       minimum=False)

    print(f"Scanparameter Röntgen: {ausbeute[0]} \n"
          f"Scanparameter Laser: {ausbeute[1]} \n"
          f"Scanparameter Farbkamera: {ausbeute[2]} \n \n")


    delaminierung = predict_linear(
                            conc_sheet=conc_sheet,
                            y_name='Delaminierung %',
                            x_names=[
                                'Lamellenbreite [mm]',
                                'Temperatur [°C]',
                                'rel. Luftfeuchtigkeit [%]',
                                'Leimfaktor',
                                'Hobelmaß Lamellenhobel Breitseite [mm]',
                                'Hobelmaß Keilzinkung Breitseite [mm]'
                            ])

    print(f"Lamellenbreite: {festigkeit[0]} \n"
          f"Temperatur: {festigkeit[1]} \n"
          f"Luftfeuchtigkeit: {festigkeit[2]} \n"
          f"Leimfaktor: {delaminierung[3]} \n"
          f"Hobelmaß Lamellenhobel Breitseite: {delaminierung[4]}\n"
          f"Hoblemaß Keilzinkung Breitseite: {delaminierung[5]} \n \n")


    hitmiss = predict_group(
                      conc_sheet=conc_sheet,
                      y_name='Hit & Miss',
                      x_names=[
                          'Lamellenbreite [mm]',
                          'Hobelmaß Lamellenhobel Breitseite [mm]',
                          'Hobelmaß Keilzinkung Breitseite [mm]',
                          'Hobelmaß Binderhobel Höhe [mm]'
                      ])

    print(f"Mögliche Einstllungen: \n"
          f"Lamellenbreite: {hitmiss[0]}\n"
          f"Hobelmaß Lamellenhobel Breitseite: {hitmiss[1]}\n"
          f"Hobelmaß Keilzinkung Breitseite: {hitmiss[2]} \n"
          f"Hobelmaß Binderhobel Höhe: {hitmiss[3]} \n \n")


    lamellenversatz = predict_group(
                              conc_sheet=conc_sheet,
                              y_name='seitl. Lamellenversatz',
                              x_names=[
                                  'Hobelmaß Keilzinkung Schmalseite [mm]',
                                  'Hobelmaß Binderhobel Breite [mm]'
                              ])


    print(f"Mögliche Einstellungen: \n"
          f"Hobelmaß Keilzinkung Schmalseite: {lamellenversatz[0]} \n"
          f"Hobelmaß Binderhobel Breite: {lamellenversatz[1]} \n \n")


    leimfugen = predict_group(
                        conc_sheet=conc_sheet,
                        y_name='offene Leimfugen',
                        x_names=[
                            'Lamellenbreite [mm]',
                            'Hobelmaß Lamellenhobel Breitseite [mm]',
                            'Hobelmaß Keilzinkung Breitseite [mm]'
                        ])

    print(f"Mögliche Einstellungen: \n"
          f"Lamellenbreite: {leimfugen[0]} \n"
          f"Hoblemaß Lamellenhobel Breitseite: {leimfugen[1]} \n"
          f"Hobelmaß Keilzinkung Breitseite: {leimfugen[2]} \n \n")
