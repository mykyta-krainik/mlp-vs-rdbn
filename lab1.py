import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('cached_datasets', exist_dok=True)


class DataLoader:
    @staticmethod
    def load_iris():
        cache_path = 'cached_datasets/iris.pkl'
        if os.path.exists(cache_path):
            print("Loading IRIS dataset from cache...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print("Downloading IRIS dataset and caching...")
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_one_hot, test_size=0.2, random_state=42
        )
        
        dataset = (X_train, y_train, X_test, y_test, iris.feature_names, iris.target_names)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset
    
    @staticmethod
    def load_mnist():
        cache_path = 'cached_datasets/mnist.pkl'
        if os.path.exists(cache_path):
            print("Loading MNIST dataset from cache...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print("Downloading MNIST dataset and caching...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        greyscale = 255.0

        X_train = X_train.astype('float32') / greyscale
        X_test = X_test.astype('float32') / greyscale
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)
        
        dataset = (X_train, y_train_one_hot, X_test, y_test_one_hot, list(range(10)))
        
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset


class ActivationFunctions:
    @staticmethod
    def identity(x):
        return x
    
    @staticmethod
    def relu(x):
        return tf.nn.relu(x)
    
    @staticmethod
    def tanh(x):
        return tf.nn.tanh(x)
    
    @staticmethod
    def gaussian(x, centers, widths):
        x = tf.cast(x, tf.float32)
        centers = tf.cast(centers, tf.float32)
        widths = tf.cast(widths, tf.float32)
        expanded_x = tf.expand_dims(x, 2)  # [batch_size, input_dim, 1]
        expanded_centers = tf.expand_dims(centers, 0)  # [1, input_dim, n_centers]
        distances = tf.reduce_sum(tf.square(expanded_x - expanded_centers), axis=1)  # [batch_size, n_centers]
        return tf.exp(-distances / (2 * tf.square(widths)))
    
    @staticmethod
    def multiquadric(x, centers, widths):
        x = tf.cast(x, tf.float32)
        centers = tf.cast(centers, tf.float32)
        widths = tf.cast(widths, tf.float32)
        expanded_x = tf.expand_dims(x, 2)
        expanded_centers = tf.expand_dims(centers, 0)
        distances = tf.reduce_sum(tf.square(expanded_x - expanded_centers), axis=1)
        return tf.sqrt(distances + tf.square(widths))


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        
        if activation == 'relu':
            activation_fn = 'relu'
        elif activation == 'tanh':
            activation_fn = 'tanh'
        elif activation == 'identity':
            activation_fn = 'linear'
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.model = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_dim, activation=activation_fn),
            layers.Dense(output_dim, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def get_hidden_output(self, X):
        hidden_model = models.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[0].output
        )
        return hidden_model.predict(X)


class RBN:
    def __init__(self, input_dim, n_centers, output_dim, rbf_type='gaussian'):
        self.input_dim = input_dim
        self.n_centers = n_centers
        self.output_dim = output_dim
        self.rbf_type = rbf_type
        
        self.centers = tf.Variable(
            tf.random.uniform([input_dim, n_centers], -1.0, 1.0),
            trainable=True,
            name="centers"
        )
        self.widths = tf.Variable(
            tf.ones([n_centers]), 
            trainable=True,
            name="widths"
        )
        
        self.W = tf.Variable(
            tf.random.normal([n_centers, output_dim], stddev=0.1),
            trainable=True,
            name="output_weights"
        )
        self.b = tf.Variable(
            tf.zeros([output_dim]),
            trainable=True,
            name="output_bias"
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    def call(self, x):
        if self.rbf_type == 'gaussian':
            rbf_output = ActivationFunctions.gaussian(x, self.centers, self.widths)
        elif self.rbf_type == 'multiquadric':
            rbf_output = ActivationFunctions.multiquadric(x, self.centers, self.widths)
        else:
            raise ValueError(f"Unsupported RBF type: {self.rbf_type}")
        
        logits = tf.matmul(rbf_output, self.W) + self.b
        return tf.nn.softmax(logits)
    
    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        )
    
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.loss_fn(y, y_pred)
        
        gradients = tape.gradient(
            loss, 
            [self.centers, self.widths, self.W, self.b]
        )
        
        self.optimizer.apply_gradients(
            zip(gradients, [self.centers, self.widths, self.W, self.b])
        )
        
        return loss, y_pred
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            indices = tf.random.shuffle(tf.range(n_samples))
            X_shuffled = tf.gather(X_train, indices)
            y_shuffled = tf.gather(y_train, indices)
            
            epoch_loss = 0
            epoch_acc = 0
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss, y_pred = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                
                y_pred_classes = tf.argmax(y_pred, axis=1)
                y_true_classes = tf.argmax(y_batch, axis=1)
                batch_acc = tf.reduce_mean(
                    tf.cast(tf.equal(y_pred_classes, y_true_classes), tf.float32)
                )
                epoch_acc += batch_acc
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            train_losses.append(epoch_loss.numpy())
            train_accs.append(epoch_acc.numpy())
            
            val_pred = self.call(X_val)
            val_loss = self.loss_fn(y_val, val_pred).numpy()
            val_losses.append(val_loss)
            
            val_pred_classes = tf.argmax(val_pred, axis=1)
            val_true_classes = tf.argmax(y_val, axis=1)
            val_acc = tf.reduce_mean(
                tf.cast(tf.equal(val_pred_classes, val_true_classes), tf.float32)
            ).numpy()
            val_accs.append(val_acc)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, "
                      f"Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accs,
            'val_accuracy': val_accs
        }
    
    def evaluate(self, X_test, y_test):
        y_pred = self.call(X_test)
        loss = self.loss_fn(y_test, y_pred).numpy()
        
        y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
        y_true_classes = tf.argmax(y_test, axis=1).numpy()
        
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }


class HybridNetwork:
    def __init__(self, mlp_model, input_dim, n_centers, output_dim, rbf_type='gaussian'):
        self.mlp_model = mlp_model
        self.hidden_dim = mlp_model.hidden_dim
        self.rbn = RBN(self.hidden_dim, n_centers, output_dim, rbf_type)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        X_train_hidden = self.mlp_model.get_hidden_output(X_train)
        X_val_hidden = self.mlp_model.get_hidden_output(X_val)
        
        return self.rbn.fit(X_train_hidden, y_train, X_val_hidden, y_val, epochs, batch_size)
    
    def evaluate(self, X_test, y_test):
        X_test_hidden = self.mlp_model.get_hidden_output(X_test)
        return self.rbn.evaluate(X_test_hidden, y_test)


def run_experiment(dataset_name, activation_functions):
    print(f"Running experiments on {dataset_name} dataset")
    
    if dataset_name == 'iris':
        X_train, y_train, X_test, y_test, feature_names, target_names = DataLoader.load_iris()
        n_classes = len(target_names)
        input_dim = X_train.shape[1]
        hidden_dim = 10
        n_centers = 8
    elif dataset_name == 'mnist':
        X_train, y_train, X_test, y_test, target_names = DataLoader.load_mnist()
        n_classes = len(target_names)
        input_dim = X_train.shape[1]
        hidden_dim = 128
        n_centers = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    results = {}
    
    for activation in activation_functions['mlp']:
        print(f"\nTraining MLP with {activation} activation")
        mlp = MLP(input_dim, hidden_dim, n_classes, activation)
        history = mlp.fit(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
        results[f'mlp_{activation}'] = {
            'model': mlp,
            'history': history,
            'eval': mlp.evaluate(X_test, y_test)
        }
    
    for rbf_type in activation_functions['rbn']:
        print(f"\nTraining RBN with {rbf_type} function")
        rbn = RBN(input_dim, n_centers, n_classes, rbf_type)
        history = rbn.fit(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
        results[f'rbn_{rbf_type}'] = {
            'model': rbn,
            'history': history,
            'eval': rbn.evaluate(X_test, y_test)
        }
    
    best_mlp_key = max(
        [k for k in results.keys() if k.startswith('mlp_')],
        key=lambda k: results[k]['eval']['accuracy']
    )
    best_mlp = results[best_mlp_key]['model']
    
    for rbf_type in activation_functions['rbn']:
        print(f"\nTraining Hybrid Network with {best_mlp_key} MLP and {rbf_type} RBN")
        hybrid = HybridNetwork(best_mlp, input_dim, n_centers, n_classes, rbf_type)
        history = hybrid.fit(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
        results[f'hybrid_{rbf_type}'] = {
            'model': hybrid,
            'history': history,
            'eval': hybrid.evaluate(X_test, y_test)
        }
    
    visualize_results(results, dataset_name)
    
    return results


def visualize_results(results, dataset_name):
    os.makedirs('results', exist_ok=True)
    
    model_names = []
    accuracies = []
    
    for model_name, model_result in results.items():
        if 'eval' in model_result:
            model_names.append(model_name)
            accuracies.append(model_result['eval']['accuracy'])
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.title(f'Model Accuracy Comparison - {dataset_name.upper()} Dataset')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_accuracy_comparison.png')
    
    for model_name, model_result in results.items():
        if 'eval' in model_result:
            plt.figure(figsize=(8, 6))
            conf_matrix = model_result['eval']['confusion_matrix']
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name} - {dataset_name.upper()}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(f'results/{dataset_name}_{model_name}_confusion_matrix.png')
    
    with open(f'results/{dataset_name}_results.txt', 'w') as f:
        for model_name, model_result in results.items():
            if 'eval' in model_result:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {model_result['eval']['accuracy']:.4f}\n")
                f.write(f"Loss: {model_result['eval']['loss']:.4f}\n")
                f.write("Classification Report:\n")
                
                report = model_result['eval']['classification_report']
                for class_name, metrics in report.items():
                    if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    f.write(f"  Class {class_name}:\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1-score: {metrics['f1-score']:.4f}\n")
                
                f.write(f"Macro Avg:\n")
                f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
                f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
                f.write(f"  F1-score: {report['macro avg']['f1-score']:.4f}\n\n")
                f.write("-" * 40 + "\n\n")


def main():
    print("Neural Network Activation Function Comparison")
    print("=" * 50)
    
    activation_functions = {
        'mlp': ['relu', 'identity', 'tanh'],
        'rbn': ['gaussian', 'multiquadric']
    }
    
    print("\n=== IRIS Dataset ===\n")
    run_experiment('iris', activation_functions)
    
    print("\n=== MNIST Dataset ===\n")
    run_experiment('mnist', activation_functions)

if __name__ == "__main__":
    main()
