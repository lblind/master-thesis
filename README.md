# Master Thesis 
## Food Security and Climate Shocks
### The Resilience of Food Prices to Droughts in Malawi. A Machine Learning Approach.

This is the GitHub repository corresponding to the Master Thesis of Lena B..

![SDG-2.png](images/sdg-2.png) ![SDG-13.png](images/sdg-13.png) 

--------------------------------------------------------------------------
## Data
The data merges data on food prices for different commodities and markets provided by WFP, as well as climate data on droughts, based on the SPEI as a drought indicator.  

**The final dataset for Malawi can be found [here](output/Malawi/Malawi-final-dta.xlsx)**.

## Structure Malawi
- [input folder Malawi](input/Malawi)
  - [food price data](input/Malawi/food-price-dta/csv-prices)
  - [lon lat for markets](input/Malawi/food-price-dta/csv-lons-and-lats)
  - [inflation data WFP](input/Malawi/inflation-dta/WFP)
  - [SPEI data*](input/Malawi/climate-dta/spei01.nc)
- [output folder Malawi](output/Malawi)
  - [Summary statistics (Doc. Preproecessing)](output/Malawi/summary-statistics)
  - [Plots](output/Malawi/plots)
- [Source code](src)
  - [main (Run this to start the program)](src/main.py)
  - [Creation of dataset](src/dataset_creation.py)
  - [Preprocessing](src/preprocessing.py)
  - [Analysis & Summary statistics](src/analysis.py)
  - [Visualization](src/visualization.py)
  - [Auxiliary funtions (utils)](src/utils.py)
  - [Stata Correlation matrices per commodity](src/corr_matrices.do)

*Hint: For the SPEI dataset, Git LFS has been used as the file size
exceeded 100 MB. It might be necessary for
you to install that Git extension as well to derive the respective file (spei01.nc) in
the [climate data folder](input/Malawi/climate-dta/).
### Food price data
![Screenshot vam.png](images/screenshot-wfp-vam.png)
[Food prices](input/Malawi/food-price-dta/csv-prices) have been obtained via the open source database of WFP.  
Temporal Unit of analysis: Month  

_Source_:

- [WFP vam database](https://dataviz.vam.wfp.org/economic_explorer/prices)

Link used to extract data for central region [14.06.2022, 14:12]:  
- [Link](https://dataviz.vam.wfp.org/economic_explorer/prices)
### Coordinates of Markets
Further data upon [market coordinates](input/Malawi/food-price-dta/csv-lons-and-lats) has been kindly provided upon request via the team of WFP.

### Climate data (drought)
![Screenshot Spei database.png](images/screenshot-spei-database.png)  
_Screenshot: [SPEI Global Drought Monitor](https://spei.csic.es/spei_database/#map_name=spei01#map_position=1439)_
- [SPEI Global Drought Monitor](https://spei.csic.es/map/maps.html#months=1#month=4#year=2022)

- [Selection Malawi](https://spei.csic.es/map/maps.html#months=0#month=4#year=2022)
- [Global SPEI Database](https://spei.csic.es/database.html#p7)  




--------------------------------------------------------------
## Additional Useful Links

### WFP
- [VAM Resource Center](https://resources.vam.wfp.org/)
- [Overview Malawi Dataviz](https://dataviz.vam.wfp.org/version2/country/malawi)
