* Outcomment the below line to import the excel sheet (after merging the datasets together)

import excel "..\output\Malawi\intermediate-results\df_wfp_w_drought_STEP4.xlsx", sheet("Sheet1") firstrow


cap encode Region, gen(RegionCat)
cap encode Market, gen(MarketCat)

cap encode Commodity, gen(CommodityCounter)
cap encode Commodity, gen(CommodityCat)

corr AdjPrice *Cat Drought Spei

codebook CommodityCounter

* Compute the correlation matrix per commodity

forvalues k=1(1)7{
	codebook Commodity if CommodityCounter == `k'
	corr AdjPrice *Cat Drought Spei if CommodityCounter == `k'
	
}

