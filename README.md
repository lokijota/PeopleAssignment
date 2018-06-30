# Team people to Customers Assignment

## Goal

I have a team of people, each located in different parts of the UK, that I want to assign to a set of customers based on criteria like travel time or strength of existing relationships.

## Implementation 

I created an Azure Databricks workspace with a small Python3 cluster. I also created an Azure Storage Account, where I deposited a file with the map of the UK. I then created a Jupyter notebook (Setup.py - not included in GitHub) which I call via a %run command from the main notebook, containing all the private variables.

The reason to include a Map instead of using solutions like embedding google/bing maps or matplotlib/basecamp, is that Databricks is not  friendly to these approaches, so I ended up having to plot on top of a pre-cut map.

The main notebook is documented. It loads the data, computes the distances (keeping a cache in dbfs to avoid exceeding Google Maps' API limits), then prints out data like a table of travel times or the locations on the map, and finally implements the Stable Marriage algorithm to do the assingments.
