import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import utility functions
from src.financial_pipeline.utils import load_csv_with_path, save_csv_with_path

# Configuration
INPUT_CSV = '../data/v9_transactions.csv'
OUTPUT_CSV = '../data/transactions_with_embeddings_enhanced.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 64

def _generate_sentence(row):
    """Generate descriptive sentence for each transaction row."""
    name = row['Name']
    party = row['Party']
    state = row['State']
    chamber = row['Chamber']
    committees = row['Committees']
    transaction = row['Transaction']
    industry = row['Industry']
    sector = row['Sector']

    sp100_status = "an S&P 100 constituent" if row['In_SP100'] else "not an S&P 100 constituent"

    return (
        f"Congressional representative {name} ({party}-{state}), serving in the {chamber} "
        f"and on the {committees} committee, executed a {transaction} transaction involving "
        f"a security from the {industry} industry within the {sector} sector. "
        f"The traded equity is {sp100_status}, representing regulatory disclosure of congressional investment activity."
    )

def enhance_embeddings(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE
):
    print("Loading transaction data...")
    df = load_csv_with_path(input_csv)

    print("Creating sentence descriptions...")
    tqdm.pandas(desc="Generating sentences")
    df['sentence'] = df.progress_apply(_generate_sentence, axis=1)

    print("\nLoading sentence transformer model...")
    model = SentenceTransformer(model_name)
    print(f"Using device: {model.device}")

    print("\nGenerating embeddings...")
    sentences = df['sentence'].tolist()
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("\nCreating embeddings DataFrame...")
    embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    df[embedding_cols] = pd.DataFrame(embeddings, index=df.index)

    print(f"\nSaving results to {output_csv}...")
    save_csv_with_path(df, output_csv, index=False)

    print(f"\nComplete! Added {embeddings.shape[1]} embedding features to {len(df)} rows.")
    print(f"Sample embeddings shape: {embeddings.shape}")
    return df