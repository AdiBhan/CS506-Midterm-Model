################################################################
# Author: Adi Bhan
# Project: CS 506 Midterm Model 
################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier ## Old Classifer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Positive and negative adjectives found in train.csv
POS_WORDS = [
            'good', 'great', 'excellent', 'amazing', 'love', 'wonderful', 'best', 'perfect',
            'fantastic', 'enjoy', 'nice', 'awesome', 'delicious', 'favorite', 'tasty',
            'satisfying', 'happy', 'pleased', 'superb', 'brilliant', 'outstanding', 'positive',
            'quality', 'spectacular', 'pleasant', 'terrific', 'beautiful', 'memorable', 'flawless',
            'charming', 'recommended', 'exciting', 'fun', 'fascinating', 'worth',
            'heartwarming', 'entertaining', 'remarkable', 'stunning', 'funny', 'exceptional', 'genius',
            'masterpiece', 'masterful', 'riveting', 'compelling', 'gripping', 'unforgettable',
            'phenomenal', 'impressive', 'delightful', 'captivating', 'powerful', 'moving',
            'touching', 'hilarious', 'witty', 'clever', 'engaging', 'refreshing', 'solid',
            'strong', 'authentic', 'inspiring', 'thoughtful', 'magical', 'elegant',
            'rich', 'intimate', 'profound', 'heavenly', 'astonishing', 'immersive',
            'engrossing', 'timeless', 'iconic', 'classic', 'revolutionary', 'innovative', 'superb',
            'excellent', 'upbeat', 'energetic', 'potent', 'classy', 'genuine', 'pure', 'honest',
            'vibrant', 'dynamic', 'sincere', 'passionate', 'mesmerizing', 'intriguing', 'haunting',
            'extraordinary', 'marvelous', 'legendary', 'sublime', 'impeccable', 'wholesome',
            'absorbing', 'delicate', 'grand', 'glorious', 'triumphant', 'stellar',
            'graceful', 'artful', 'skillful', 'exhilarating', 'bold', 'radiant',
            'brave', 'fearless', 'generous', 'charismatic', 'breathtaking', 'mind-blowing',
            'tour-de-force', 'nuanced', 'spellbinding', 'groundbreaking', 'poignant', 'virtuoso',
            'immaculate', 'polished', 'seamless', 'enthralling', 'dazzling', 'transcendent',
            'electrifying', 'visionary', 'robust', 'infectious', 'pitch-perfect', 'razor-sharp',
            'masterclass', 'top-notch', 'evocative', 'accomplished', 'soul-stirring', 'crystalline',
            'arresting', 'revelatory', 'unrivaled', 'pristine', 'luminous', 'formidable',
            'resplendent', 'impactful', 'assured', 'consummate', 'first-rate', 'nimble',
            'laudable', 'incisive', 'splendid', 'sumptuous', 'harmonious', 'perspicacious',
            'kaleidoscopic', 'inventive', 'ravishing', 'textured', 'audacious', 'taut',
            'hypnotic', 'penetrating'
        ]
NEG_WORDS = [
            'bad', 'terrible', 'awful', 'worst', 'poor', 'horrible', 'waste', 'disappoint',
            'boring', 'hate', 'bland', 'disgusting', 'not good', 'unfortunately', 'gross',
            'mediocre', 'annoying', 'frustrating', 'unpleasant', 'cheap', 'dull', 'forgettable',
            'dissatisfied', 'lacking', 'predictable', 'flawed', 'dislike', 'disappointing',
            'uninspired', 'underwhelming', 'slow', 'cringe', 'poorly', 'unconvincing', 'stereotypical',
            'atrocious', 'unwatchable', 'mess', 'disaster', 'garbage', 'abysmal',
            'nonsensical', 'pointless', 'tedious', 'pathetic', 'ridiculous', 'shallow',
            'lazy', 'amateurish', 'incoherent', 'confusing', 'stupid', 'dreadful',
            'tiresome', 'weak', 'flat', 'lackluster', 'empty', 'incompetent', 'lifeless',
            'miserable', 'stiff', 'awkward', 'distant', 'unlikable', 'self-absorbed', 'gimmicky',
            'dragging', 'muffled', 'incoherent', 'schizophrenic', 'herky-jerky', 
            'jarring', 'hackneyed', 'clichéd', 'paint-by-numbers', 'wooden', 
            'dopey', 'cheesy', 'disjointed', 'implausible', 'underwhelming', 
            'far-fetched', 'sluggish', 'plodding', 'stale', 'pretentious', 'grotesque', 
            'gruesome', 'monotonous', 'derivative', 'unremarkable', 'subpar', 
            'half-baked', 'contrived', 'convoluted', 'muddled', 'tiresome', 'overrated',
            'predictable', 'mediocre', 'cliché', 'dismal', 'unfocused', 'exhausting', 'cringe-worthy', 'rushed',
            'unrealistic', 'excessive', 'ham-fisted', 'tone-deaf', 'bloated', 'soulless', 'hamstrung',
            'meandering', 'half-hearted', 'formulaic', 'turgid', 'schlocky', 'sophomoric', 'vapid',
            'heavy-handed', 'insipid', 'middling', 'banal', 'clunky', 'mawkish', 'preachy',
            'melodramatic', 'long-winded', 'by-the-numbers', 'superficial', 'outdated', 'grating',
            'miscast', 'overwrought', 'stilted', 'farcical', 'mind-numbing', 'heavy-going',
            'two-dimensional', 'laborious', 'jumbled', 'watered-down', 'rambling', 'pedestrian',
            'uninvolving', 'kitsch', 'patronizing', 'dragged-out'
        ]
# Features used in the model
FEATURES = [
    'Time', 'HelpfulnessDenominator', 'PosAdjCount', 'NegAdjCount',
    'ReviewLen', 'ExMarkCount', 'QuesMarkCount', 'AvgWordLen',
    'UniqueWordCount', 'HelpfulnessRatio', 'Year', 'PosNegRatio', 'WordDiversity', 
    'ComplexityScore', 'PolarityScore', 
    'SubjectivityScore'
] 

def setup_env(seed=42):
    """setup_env sets up the environment using random and seeds so code produces consistent results each time."""
    np.random.seed(seed)
    random.seed(seed)
    return seed
def load_csv_dataset():
    ''' load_dat reads data from train and test csv files'''
    print('---------------------------------------------------------------'
    )
    print("Loading datasets...")
    train_df = pd.read_csv("./data/train.csv", nrows=2000)
    test_df = pd.read_csv("./data/test.csv", nrows=2000)

    print("train.csv shape is ", train_df.shape)
    print("test.csv shape is ", test_df.shape)
    # Display some basic information
    print("\nTraining Set Preview:\n", train_df.head())
    print("\nTesting Set Preview:\n", test_df.head())
    print("\nTraining Set Description:\n", train_df.describe())

    # Plotting the distribution of the target variable in the training set
    print("Plotting distribution of 'Score' in training set...")
    train_df['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
    print("\nEVERYTHING IS PROPERLY SET UP! YOU ARE READY TO START\n")
    print('---------------------------------------------------------------\nFINISHED LOADING!'
    )
    
    return train_df, test_df
def add_features_to(df):
    '''add_features_to adds features to the training set to help Model predict movie review scores'''
    
    print("Adding Features to DataFrame...")
    print('---------------------------------------------------------------'
    )
    # Clean missing values
    df['HelpfulnessDenominator'] = (df['HelpfulnessDenominator']
                                   .replace(0, np.nan)
                                   .fillna(0))
    df['Text'] = df['Text'].fillna('')
    # Sentiment Scores using TextBlob
    df['PolarityScore'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['SubjectivityScore'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    # Count of positive and negative words
    df['PosAdjCount'] = df['Text'].apply(
        lambda x: sum(x.lower().count(word) for word in POS_WORDS)
    )
    df['NegAdjCount'] = df['Text'].apply(
        lambda x: sum(x.lower().count(word) for word in NEG_WORDS)
    )
    df['ReviewLen'] = df['Text'].apply(lambda x: len(x.split()))
    df['ExMarkCount'] = df['Text'].apply(lambda x: x.count('!'))
    df['QuesMarkCount'] = df['Text'].apply(lambda x: x.count('?'))
    # -	Text Complexity and Diversity Features 
    df['AvgWordLen'] = df['Text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    df['UniqueWordCount'] = df['Text'].apply(lambda x: len(set(x.split())))
    df['WordDiversity'] = df['UniqueWordCount'] / (df['ReviewLen'] + 1)
    df['ComplexityScore'] = (df['AvgWordLen'] * df['UniqueWordCount'] / 
                            (df['ReviewLen'] + 1))
    df['PosNegRatio'] = df['PosAdjCount'] / (df['NegAdjCount'] + 1)
    df['HelpfulnessRatio'] = df.apply(
        lambda row: (row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] 
                    if row['HelpfulnessDenominator'] > 0 else 0),
        axis=1
    )
    df['Year'] = pd.to_datetime(df['Time'], unit='s').dt.year
    return df
def prepare_data(train_df, test_df):
    ''' Prepares the training and test datasets by processing training data
    and matching it with test data based on 'Id'.'''
    
    train_df = add_features_to(train_df)

    test_data = pd.DataFrame()
    for test_id in test_df['Id']:
        matching_row = train_df[train_df['Id'] == test_id]
        if not matching_row.empty:
            test_data = pd.concat([test_data, matching_row])
    return test_data
def handle_missing_entries(train_df, test_data, sample_size=500):
    ''' If no matching IDs are found, create a random sample of 500 records/rows from the training data '''
    if test_data.empty:
        print("No matching IDs found in training data. CAdding missing entries...")
        # Use a portion of training data as test data
        train_data = train_df[train_df['Score'].notna()].copy()
        test_indices = random.sample(list(train_data.index), min(500, len(train_data)))
        test_data = train_data.loc[test_indices].copy()
        train_df = train_df.drop(test_indices)
    return train_df, test_data
def find_correlation_features(train_df, features):
    ''' find_correlation_features helper function logs the correlation of each feature against the score '''
    print("\n--------------------------------------------------")
    correlation_with_score = train_df[features + ['Score']].corr()['Score'].drop('Score') * 100
    correlation_with_score = correlation_with_score.round(2)
    print("Correlation of each feature with 'Score' (in %):")
    print(correlation_with_score)
    print("\n--------------------------------------------------")
def plot_feature_correlations(train_df, features):
    ''' plot_feature_correlations function graphs Feature Correlation with Review Score for first 10,000 records'''
    plt.figure(figsize=(15, 8))
    correlations = train_df[features + ['Score']].corr()['Score'].drop('Score').sort_values(ascending=True)
    colors = ['red' if x < 0 else 'green' for x in correlations.values]
    sns.barplot(x=correlations.values, y=correlations.index, palette=colors, hue=correlations.index, dodge=False, legend=False)
    
    # Updated title to mention the dataset size
    plt.title('Feature Correlations with Review Score (First 10,000 Records)', fontsize=14, pad=20)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    os.makedirs('graphs', exist_ok=True)
    plt.savefig("graphs/feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, features):
    ''' plot_feature_importance() function graphs Random Forest Importance for first 10,000 records'''
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        
        plt.title('Random Forest Feature Importance (First 10,000 Records)', fontsize=14, pad=20)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
       
        os.makedirs('graphs', exist_ok=True)
        plt.savefig("graphs/feature_importance.png", dpi=300)
        plt.show()
    else:
        print("Model does not have feature importances.")
        
def test_model(Y_val, Y_val_predictions):
    ''' test_model() evaluates the accuracy and generates a confusion matrix plot'''
    print("Accuracy on validation set =", accuracy_score(Y_val, Y_val_predictions))
    # Confusion Matrix
    print("Generating confusion matrix...")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(Y_val, Y_val_predictions, normalize='true')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix of the Random Forest Classifier for first 10,000 records')
    plt.xlabel('Predicted')
    plt.ylabel('True')    
    os.makedirs('graphs', exist_ok=True)
    plt.savefig("graphs/confusion_matrix.png", dpi=300)
    plt.show()


def main():
    seed = setup_env()
    train_df, test_df = load_csv_dataset()
    test_data = prepare_data(train_df, test_df)
    
    # Process train_df
    print("Processing training data...")
    train_df = add_features_to(train_df)

    # Handle missing entries
    train_df, test_data = handle_missing_entries(train_df, test_data)
    print("Selected features:", FEATURES)

    # Find correlation of features with Score
    find_correlation_features(train_df, FEATURES)

    # Prepare features and target
    train_data = train_df[train_df['Score'].notna()]
    X = train_data[FEATURES]
    y = train_data['Score']

    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    print("\n--------------------------------------------------------")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("--------------------------------------------------------\n")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_data[FEATURES])

    # Using RandomForestClassifier
    print("Training the model...")
    model = RandomForestClassifier(
        n_estimators=700, 
        max_depth=20,
        min_samples_split=4,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train_scaled, Y_train)

    # Evaluate on validation set
    print("Evaluating the model...")
    Y_val_predictions = model.predict(X_val_scaled)

    test_model(Y_val, Y_val_predictions)

    # Predict on test data
    print("Making predictions on the test data...")
    test_predictions = model.predict(X_test_scaled)

    # Create submission file
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'Score': test_predictions
    })
    submission.to_csv("./data/submission.csv", index=False)
    print("Submission file saved as './data/submission.csv'")
    print("--------------------------------------------------------")
    print("Creating Graphs for reports...")
    plot_feature_correlations(train_df, FEATURES)
    plot_feature_importance(model, FEATURES)
    
if __name__ == "__main__":
    main()
