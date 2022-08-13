* clear

* Change to folder where you have cloned the GitHub repo
cd "C:\Users\Lena\PycharmProjects\master-thesis\"
* Import the excel sheet
import excel ".\output\Malawi\intermediate-results\df_wfp_w_drought_STEP4.xlsx", sheet("Sheet1") firstrow


cap encode Region, gen(RegionCat)
cap encode Market, gen(MarketCat)

cap encode Commodity, gen(CommodityCounter)
cap encode Commodity, gen(CommodityCat)

corr AdjPrice *Cat Drought Spei

codebook CommodityCounter

* Compute the correlation matrix per commodity 
* (in this case run k until 7 as 7 unique commodities for Malawi)
* change this is you run it for another country (7 = # unique commodities)

forvalues k=1(1)7{
	codebook Commodity if CommodityCounter == `k'
	corr AdjPrice *Cat Drought Spei if CommodityCounter == `k'
	
}

