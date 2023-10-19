import great_expectations as ge
import pandas as pd

context = ge.get_context()

context.add_or_update_expectation_suite("speech_commands_suite")

datasource = context.sources.add_or_update_pandas(name="speech_commands_dataset")

# DATAFRAME DEFINITION
# train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# validation = pd.read_csv("validation.csv")
df = ge.dataset.PandasDataset(test)
# df = pd.concat([train, test, validation])


data_asset = datasource.add_dataframe_asset(name="speech_commands_asset", dataframe=df)

batch_request = data_asset.build_batch_request()
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="speech_commands_suite",
    datasource_name="speech_commands_dataset",
    data_asset_name="speech_commands_asset",
)

validator.expect_table_columns_to_match_ordered_list(
    column_list=[
        "file",
        "audio",
        "label",
        "is_unknown",
        "speaker_id",
        "utterance_id"
    ]
)

validator.expect_column_values_to_be_unique("file")
validator.expect_column_values_to_not_be_null("file")
validator.expect_column_values_to_be_of_type("file", "str")

validator.expect_column_values_to_be_unique("audio")
validator.expect_column_values_to_not_be_null("audio")

validator.expect_column_values_to_not_be_null("label")
validator.expect_column_values_to_be_of_type("label", "int64")

validator.save_expectation_suite(discard_failed_expectations=False)

checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)
