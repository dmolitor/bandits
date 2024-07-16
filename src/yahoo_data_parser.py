import gzip
import pandas as pd
from tqdm import tqdm


# Function to parse a single line of data and return two chunks
def parse_line(line: str):
    parts = line.split()
    timestamp = int(parts[0])
    displayed_article_id = int(parts[1])
    user_click = int(parts[2])

    user_features = {}
    article_features = {}

    parsing_user_features = True
    parsing_article_id = None

    for part in parts[3:]:
        if part == "|user":
            parsing_user_features = True
            continue
        elif part.startswith("|"):
            parsing_user_features = False
            parsing_article_id = int(part[1:])
            article_features[parsing_article_id] = {}
            continue

        if parsing_user_features:
            feature_id, feature_value = part.split(":")
            user_features[f"user_feature_{feature_id}"] = float(feature_value)
        elif parsing_article_id is not None:
            feature_id, feature_value = part.split(":")
            article_features[parsing_article_id][f"article_feature_{feature_id}"] = (
                float(feature_value)
            )

    return {
        "timestamp": timestamp,
        "displayed_article_id": displayed_article_id,
        "user_click": user_click,
        "user_features": user_features,
        "article_features": article_features,
    }


class YahooDataParser:
    def __init__(self, path: str):
        self.current_record = 0
        with gzip.open(path, "rt") as f:
            records = f.readlines()
        self.records = records
        self.n_records = len(records)

    def read_line(self, line_number: int) -> str | None:
        if line_number > (self.n_records - 1):
            raise ValueError(f"Invalid record index: {line_number}")
        return self.records[line_number]

    def next_record(self) -> "YahooDataRecord":
        line = self.read_line(line_number=self.current_record)
        parsed_line = parse_line(line)
        data_record = YahooDataRecord(parsed_line)
        self.current_record += 1
        return data_record


class YahooDataRecord:
    def __init__(self, record: dict | None):
        if record is None:
            self.user_data = None
            self.article_data = None
            return None
        user_data = {
            "timestamp": record["timestamp"],
            "displayed_article_id": record["displayed_article_id"],
            "user_click": record["user_click"],
            **record["user_features"],
        }
        user_df = pd.DataFrame([user_data])
        article_df = pd.DataFrame(record["article_features"]).T.reset_index()
        article_df.columns = ["article_id"] + list(article_df.columns[1:])
        self.user_data = user_df
        self.article_data = article_df


if __name__ == "__main__":
    print("Importing data ...")
    data_parser = YahooDataParser(
        path="/Users/dmolitor/Downloads/R6/ydata-fp-td-clicks-v1_0.20090501.gz"
    )
    print("Iterating through 10,000 records")
    for _ in tqdm(range(10000)):
        row = data_parser.next_record()
