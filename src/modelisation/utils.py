def create_labels(data, n_labels):
    labels = [[] for i in range(len(data))]
    labels_names = list(data.columns[:n_labels])
    for i in range(len(data)):
        vector = list(data.iloc[i][:n_labels])
        for j, elt in enumerate(vector):
            if elt:
                labels[i].append(labels_names[j])
    return labels


def transform_labels_into_strings(L):
    string = ''
    for elt in L:
        if len(string) == 0:
            string += elt
        else:
            string += '/' + elt
    return string


def count_label_occurences(L):
    count = 0
    values = []
    freq = {}

    for x in L:
        label = transform_labels_into_strings(x)
        if x in values:
            freq[label] += 1
        else:
            freq[label] = 1
            values.append(x)

    for key, value in freq.items():
        if value == 1:
            count += 1

    return count, freq


def keep_unique_label_data(df):
    df_unique = df
    to_drop = []
    labels = create_labels(df_unique, 21)
    for i in range(len(labels)):
        if len(labels[i]) > 1:
            to_drop.append(i)
    df_unique.drop(to_drop, 0, inplace=True)
    return df_unique


def convert_string_labels_into_numbers(df, labels):
    genres = list(df.columns[:21])
    Y = [genres.index(label[0]) for label in labels]
    return Y
