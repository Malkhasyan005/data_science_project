import src.data_processing.data_loader as data_loader

if __name__ == "__main__":
    loader = data_loader.FreshRetailDataLoader()
    train_df, eval_df = loader.load_data()
    print("Data loaded successfully")
    print(f"Train Data Shape: {train_df.shape}")
    print(f"Eval Data Shape: {eval_df.shape}")
    print(loader.get_data_summary())