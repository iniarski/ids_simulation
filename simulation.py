import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

import model_data

default_config = {
    'n_samples': 1000000,
    'attack_rate': 0.5,
    'detection_threshold': 0.5,
    'n_classes': 6,
    'attacks_distribution': [1.0, 1.0, 1.0, 1.0, 1.0],
}

example_models = [
    model_data.binary95,
    model_data.binary97,
    model_data.multiclass98,
]

def print_classification_metrics(y_true, y_pred, average='binary'):

    # Compute the metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def is_binary_model(model : dict) -> bool:
    return model['n_classes'] == 2

def get_model_weight(model : dict) -> float:
    avg_recall = sum(model['recalls']) / model['n_classes']

    return 1 / (1 - avg_recall)

def generate_sample(attack_rate : float, attack_distribution : list[float]) -> int:
    if random.random() > attack_rate:
        return 0
    
    attack = random.random()
    cum_prob = 0

    for i, attack_prob in enumerate(attack_distribution):
        cum_prob += attack_prob
        if cum_prob > attack:
            return i + 1

def get_model_output(p : float, sample_x : float) -> float:
    prediction_offset = (1.0 - p) * random.random()
    if prediction_offset < sample_x < prediction_offset + p:
        midpoint = prediction_offset + 0.5 * p
        d = abs(midpoint - sample_x)
        model_output = math.sqrt(math.sin(d * math.pi / p))
        return (model_output + 1.0) / 2.0 # shifting from [-1, 1] to [0, 1]
    if (sample_x < prediction_offset):
        d = prediction_offset - sample_x
        model_output = -math.sqrt(math.sin(d * math.pi / prediction_offset))
        return (model_output + 1.0) / 2.0
    # sample_x > output_offset + p
    d = sample_x - prediction_offset - p
    model_output = -math.sqrt(math.sin(d * math.pi / (1.0 - p - prediction_offset)))
    return (model_output + 1.0) / 2.0

def get_adjusted_model_output(p : float, sample_x : float, inverse : bool) -> float:
    if inverse:
        return 1 - get_model_output(p, sample_x)
    else:
        return get_model_output(p, sample_x)

def get_binary_outputs(binary_models : list[dict], sample : int, sample_x : float) -> list[float]:
    bin_sample = 0 if sample == 0 else 1
    binary_outputs = list(map(lambda model: get_adjusted_model_output(model['recalls'][bin_sample], sample_x, bin_sample == 0),
                              binary_models))
    return binary_outputs

def get_multicass_model_output(model : dict, sample : int, sample_x : float) -> list[float]: 
    return [
        get_adjusted_model_output(model['recalls'][sample], sample_x, sample != i + 1) for i in range(model['n_classes'] - 1)
    ]

def get_multiclass_outputs(multiclass_models : list[dict], sample : int, sample_x : float) -> list[list[float]]:
    multiclass_outputs = [
        get_multicass_model_output(model, sample, sample_x) for model in multiclass_models
    ]
    return multiclass_outputs

def simulate(models : list[dict], config: dict = default_config, print_outputs : bool = False, weighted_avg : bool = True, show_plot : bool = False) -> None:
    n_samples = config['n_samples']
    attack_rate = config['attack_rate']
    detection_threshold = config['detection_threshold']
    n_classes = config['n_classes']
    attack_distribution = [ad / sum(config['attacks_distribution']) for ad in config['attacks_distribution']]
    
    binary_models = []
    multiclass_models = []
    binary_weights = []
    multiclass_weights = []

    for model in models:
        if is_binary_model(model):
            binary_models.append(model)
        else:
            multiclass_models.append(model)

    if len(binary_models) > 0:
        if weighted_avg:
            binary_weights = list(map(lambda model : get_model_weight(model), binary_models))
            weights_sum = sum(binary_weights)
            binary_weights = [weight / weights_sum for weight in binary_weights]
        else:
            binary_weights = len(binary_models) * [ 1 / len(binary_models)]

    if len(multiclass_models) > 0:
        if weighted_avg:
            multiclass_weights = list(map(lambda model : get_model_weight(model), multiclass_models))
            weights_sum = sum(multiclass_weights)
            multiclass_weights = [weight / weights_sum for weight in multiclass_weights]
        else:
            multiclass_weights = len(multiclass_models) * [ 1 / len(multiclass_models)]

    samples = [generate_sample(attack_rate, attack_distribution) for _ in range(n_samples)]
    samples_xs = [random.random() for _ in range(n_samples)]

    if len(binary_models) > 0:
        binary_outputs = list(map(lambda sample, sample_x : get_binary_outputs(binary_models, sample, sample_x), samples, samples_xs))
    
    if len(multiclass_models) > 0:
        multiclass_outputs = list(map(lambda sample, sample_x : get_multiclass_outputs(multiclass_models, sample, sample_x), samples, samples_xs))

    if print_outputs:
        for sample, bin_out, mult_out in zip(samples, binary_outputs, multiclass_outputs):
            print(f"Sample {sample}, binary outputs {bin_out}, multiclass_outputs:")
            for m in mult_out:
                print(m)

    binary_qualities = list(map(lambda model: sum(model['recalls']), binary_models))
    multiclass_qualities = list(map(lambda model: sum(model['recalls']), multiclass_models))
    best_binary_id = binary_qualities.index(max(binary_qualities))
    best_multiclass_id = multiclass_qualities.index(max(multiclass_qualities))

    combined_binary_predictions = n_samples * [0]
    combined_multiclass_predictions = n_samples * [0]
    best_binary_predictions = n_samples * [0]
    best_multiclass_predictions = n_samples * [0]

    for i, (sample, bin_out, mult_out) in enumerate(zip(samples, binary_outputs, multiclass_outputs)):
        weighted_bin_out = 0
        for bo, weight in zip(bin_out, binary_weights):
            weighted_bin_out += bo * weight
        
        combined_binary_predictions[i] = 1 if weighted_bin_out > detection_threshold else 0
        best_binary_predictions[i] = 1 if bin_out[best_binary_id] > detection_threshold else 0

        if combined_binary_predictions[i] == 0:
            combined_multiclass_predictions[i] == 0
        else:
            weighted_mult_out = (n_classes - 1) * [0]
            for mo, weight in zip(mult_out, multiclass_weights):
                for j, m in enumerate(mo):
                    weighted_mult_out[j] += m * weight

            max_mult = max(weighted_mult_out)
            combined_multiclass_predictions[i] = weighted_mult_out.index(max_mult) + 1

        
        if best_binary_predictions[i] == 0:
            best_multiclass_predictions[i] = 0
        else:
            best_multiclass = mult_out[best_multiclass_id]
            max_best_multiclass = max(best_multiclass)    
            best_multiclass_predictions[i] = best_multiclass.index(max_best_multiclass) + 1
    
    binarized_samples = list(map(lambda sample : 0 if sample == 0 else 1, samples))

    ConfusionMatrixDisplay.from_predictions(binarized_samples, combined_binary_predictions)
    print("Combined binary metrics")
    print_classification_metrics(binarized_samples, combined_binary_predictions)
    if show_plot:
        plt.show()
    ConfusionMatrixDisplay.from_predictions(binarized_samples, best_binary_predictions)
    print("Best binary metrics")
    print_classification_metrics(binarized_samples, best_binary_predictions)
    if show_plot:
        plt.show()
    ConfusionMatrixDisplay.from_predictions(samples, combined_multiclass_predictions)
    print("Combined multiclass metrics")
    print_classification_metrics(samples, combined_multiclass_predictions, average='macro')
    if show_plot:
        plt.show()
    ConfusionMatrixDisplay.from_predictions(samples, best_multiclass_predictions)
    print("Best multiclass metrics")
    print_classification_metrics(samples, best_multiclass_predictions, average='macro')
    if show_plot:
        plt.show()

def main():
    simulate(example_models, print_outputs=False, weighted_avg=False)

if __name__ == '__main__':
    main()