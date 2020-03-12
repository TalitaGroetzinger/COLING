## Overview 
This repository is used for the CoLing paper. 

## Current Procedure to get data 
- Use get_pos_tags.py to add a tag ['Differences'] to the wikihow_instance. 
- The script get_db_stats.py can be used to check how many of those ``wikihow_instance['Differences']`` occur in PPDB. 
- process_data_for_clf.py is currently used to add the entailment relations as a list to ``wikihow_instance['Entailment_REL']``. 
- get_dev_test_sats.py can be used to add the key wikihow_instance['Loc_in_splits'] to mark if the wikihow_instance occurs in train,dev or test. 
