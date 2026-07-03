A rule-based classification model that identifies wild mushrooms as either safe/edible or poisonous based on physical nominal attributes.

Core Architecture -
*   ML Task: Binary Classification
*   Model Implemented: Decision Tree Classifier (`max_depth=5`)
*   Data Preprocessing: Nominal One-Hot Encoding (`drop='first'`) to completely avoid false category hierarchies.

Because a False Negative (predicting a poisonous mushroom is edible) is catastrophic, this model optimization focuses heavily on minimizing the bottom-left quadrant of the Confusion Matrix to ensure maximum public safety.
