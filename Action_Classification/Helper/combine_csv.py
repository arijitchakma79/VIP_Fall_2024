import pandas as pd

class Helper:
    def combine_cvs(self, output_file_path, *files_paths):
        
        if not files_paths:
            print("No files were provided")
            return pd.DataFrame()

        combined_df = pd.concat([pd.read_csv(file) for file in files_paths], ignore_index=True)
        
        combined_df.to_csv(output_file_path, index=False)
        
        print(f"Combined CSV file saved to {output_file_path}")
        return None

if __name__ == '__main__':
    helper = Helper()
    file_1 = '../Files/queues-3actions.csv'
    file_2 = '../Files/queues-5actions.csv'
    file_3 = '../Files/queues-7actions.csv'
    output_file = '../Files/combined.csv'

    helper.combine_cvs(output_file, file_1, file_2, file_3)