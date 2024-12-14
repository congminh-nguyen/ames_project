---
Link: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt
---

## Population
SIZE: 2930 observations, 82 variables

## Article Title
Ames Iowa: Alternative to the Boston Housing Data Set

## Descriptive Abstract
Data set contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010.

## Sources
Ames, Iowa Assessor’s Office

## Variable Descriptions
Tab characters are used to separate variables in the data file. The data has 82 columns which include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables (and 2 additional observation identifiers).

- **Order (Discrete)**: Observation number
- **PID (Nominal)**: Parcel identification number - can be used with city web site for parcel review

### Sale Details
- **SalePrice (Continuous)**: The property's sale price in dollars (target variable)
- **SaleType (Nominal)**: Type of sale:
  - `WD`: Warranty Deed - Conventional
  - `CWD`: Warranty Deed - Cash
  - `VWD`: Warranty Deed - VA Loan
  - `New`: Home just constructed and sold
  - `COD`: Court Officer Deed/Estate
  - `Con`: Contract 15% Down payment regular terms
  - `ConLw`: Contract Low Down payment and low interest
  - `ConLI`: Contract Low Interest
  - `ConLD`: Contract Low Down
  - `Oth`: Other
- **SaleCondition (Nominal)**: Condition of sale:
  - `Normal`: Normal Sale
  - `Abnorml`: Abnormal Sale - trade, foreclosure, short sale
  - `AdjLand`: Adjoining Land Purchase
  - `Alloca`: Allocation - two linked properties with separate deeds
  - `Family`: Sale between family members
  - `Partial`: Home not completed when last assessed

### Property Information
- **MSSubClass (Nominal)**: The building class:
  - `020`: 1-STORY 1946 & NEWER ALL STYLES
  - `030`: 1-STORY 1945 & OLDER
  - `040`: 1-STORY W/FINISHED ATTIC ALL AGES
  - `045`: 1-1/2 STORY - UNFINISHED ALL AGES
  - `050`: 1-1/2 STORY FINISHED ALL AGES
  - `060`: 2-STORY 1946 & NEWER
  - `070`: 2-STORY 1945 & OLDER
  - `075`: 2-1/2 STORY ALL AGES
  - `080`: SPLIT OR MULTI-LEVEL
  - `085`: SPLIT FOYER
  - `090`: DUPLEX - ALL STYLES AND AGES
  - `120`: 1-STORY PUD (Planned Unit Development) - 1946 & NEWER
  - `150`: 1-1/2 STORY PUD - ALL AGES
  - `160`: 2-STORY PUD - 1946 & NEWER
  - `180`: PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
  - `190`: 2 FAMILY CONVERSION - ALL STYLES AND AGES
- **MSZoning (Nominal)**: Identifies the general zoning classification of the sale:
  - `A`: Agriculture
  - `C`: Commercial
  - `FV`: Floating Village Residential
  - `I`: Industrial
  - `RH`: Residential High Density
  - `RL`: Residential Low Density
  - `RP`: Residential Low Density Park
  - `RM`: Residential Medium Density
- **LotFrontage (Continuous)**: Linear feet of street connected to property (NA = missing data)
- **LotArea (Continuous)**: Lot size in square feet
- **Street (Nominal)**: Type of road access to property:
  - `Grvl`: Gravel
  - `Pave`: Paved
- **Alley (Nominal)**: Type of alley access to property:
  - `Grvl`: Gravel
  - `Pave`: Paved
  - `NA`: No alley access
- **LotShape (Ordinal)**: General shape of property:
  - `Reg`: Regular
  - `IR1`: Slightly irregular
  - `IR2`: Moderately Irregular
  - `IR3`: Irregular
- **LandContour (Nominal)**: Flatness of the property:
  - `Lvl`: Near Flat/Level
  - `Bnk`: Banked - Quick and significant rise from street grade to building
  - `HLS`: Hillside - Significant slope from side to side
  - `Low`: Depression
- **Utilities (Ordinal)**: Type of utilities available:
  - `AllPub`: All public Utilities (E,G,W,& S)
  - `NoSewr`: Electricity, Gas, and Water (Septic Tank)
  - `NoSeWa`: Electricity and Gas Only
  - `ELO`: Electricity only
- **LotConfig (Nominal)**: Lot configuration:
  - `Inside`: Inside lot
  - `Corner`: Corner lot
  - `CulDSac`: Cul-de-sac
  - `FR2`: Frontage on 2 sides of property
  - `FR3`: Frontage on 3 sides of property
- **LandSlope (Ordinal)**: Slope of property:
  - `Gtl`: Gentle slope
  - `Mod`: Moderate Slope
  - `Sev`: Severe Slope
- **Neighborhood (Nominal)**: Physical locations within Ames city limits:
  - `Blmngtn`: Bloomington Heights
  - `Blueste`: Bluestem
  - `BrDale`: Briardale
  - `BrkSide`: Brookside
  - `ClearCr`: Clear Creek
  - `CollgCr`: College Creek
  - `Crawfor`: Crawford
  - `Edwards`: Edwards
  - `Gilbert`: Gilbert
  - `Greens`: Greens
  - `GrnHill`: Green Hills
  - `IDOTRR`: Iowa DOT and Rail Road
  - `Landmrk`: Landmark
  - `MeadowV`: Meadow Village
  - `Mitchel`: Mitchell
  - `Names`: North Ames
  - `NoRidge`: Northridge
  - `NPkVill`: Northpark Villa
  - `NridgHt`: Northridge Heights
  - `NWAmes`: Northwest Ames
  - `OldTown`: Old Town
  - `SWISU`: South & West of Iowa State University
  - `Sawyer`: Sawyer
  - `SawyerW`: Sawyer West
  - `Somerst`: Somerset
  - `StoneBr`: Stone Brook
  - `Timber`: Timberland
  - `Veenker`: Veenker
- **Condition1 (Nominal)**: Proximity to various conditions:
  - `Artery`: Adjacent to arterial street
  - `Feedr`: Adjacent to feeder street
  - `Norm`: Normal
  - `RRNn`: Within 200' of North-South Railroad
  - `RRAn`: Adjacent to North-South Railroad
  - `PosN`: Near positive off-site feature--park, greenbelt, etc.
  - `PosA`: Adjacent to positive off-site feature
  - `RRNe`: Within 200' of East-West Railroad
  - `RRAe`: Adjacent to East-West Railroad
- **Condition2 (Nominal)**: Proximity to various conditions (if more than one is present):
  - `Artery`: Adjacent to arterial street
  - `Feedr`: Adjacent to feeder street
  - `Norm`: Normal
  - `RRNn`: Within 200' of North-South Railroad
  - `RRAn`: Adjacent to North-South Railroad
  - `PosN`: Near positive off-site feature--park, greenbelt, etc.
  - `PosA`: Adjacent to positive off-site feature
  - `RRNe`: Within 200' of East-West Railroad
  - `RRAe`: Adjacent to East-West Railroad

### Building Details
#### Dwelling Type
- **BldgType (Nominal)**: Type of dwelling:
  - `1Fam`: Single-family Detached
  - `2FmCon`: Two-family Conversion; originally built as one-family dwelling
  - `Duplx`: Duplex
  - `TwnhsE`: Townhouse End Unit
  - `TwnhsI`: Townhouse Inside Unit
- **HouseStyle (Nominal)**: Style of dwelling:
  - `1Story`: One story
  - `1.5Fin`: One and one-half story: 2nd level finished
  - `1.5Unf`: One and one-half story: 2nd level unfinished
  - `2Story`: Two story
  - `2.5Fin`: Two and one-half story: 2nd level finished
  - `2.5Unf`: Two and one-half story: 2nd level unfinished
  - `SFoyer`: Split Foyer
  - `SLvl`: Split Level

#### Overall Ratings (Aggregated)
- **OverallQual (Ordinal)**: Overall material and finish quality:
  - `10`: Very Excellent
  - `9`: Excellent
  - `8`: Very Good
  - `7`: Good
  - `6`: Above Average
  - `5`: Average
  - `4`: Below Average
  - `3`: Fair
  - `2`: Poor
  - `1`: Very Poor
- **OverallCond (Ordinal)**: Overall condition rating:
  - `10`: Very Excellent
  - `9`: Excellent
  - `8`: Very Good
  - `7`: Good
  - `6`: Above Average
  - `5`: Average
  - `4`: Below Average
  - `3`: Fair
  - `2`: Poor
  - `1`: Very Poor

#### Timing of Building (General)
- **YearBuilt (Discrete)**: Original construction date
- **YearRemodAdd (Discrete)**: Remodel date (same as construction date if no remodeling or additions)
- **GarageYrBlt (Discrete)**: Year garage was built (NA = no garage)

#### Roof Details (Specific)
- **RoofStyle (Nominal)**: Type of roof:
  - `Flat`: Flat
  - `Gable`: Gable
  - `Gambrel`: Gambrel (Barn)
  - `Hip`: Hip
  - `Mansard`: Mansard
  - `Shed`: Shed
- **RoofMatl (Nominal)**: Roof material:
  - `ClyTile`: Clay or Tile
  - `CompShg`: Standard (Composite) Shingle
  - `Membran`: Membrane
  - `Metal`: Metal
  - `Roll`: Roll
  - `Tar&Grv`: Gravel & Tar
  - `WdShake`: Wood Shakes
  - `WdShngl`: Wood Shingles

#### Exterior Details (Specific)
- **Exterior1st (Nominal)**: Exterior covering on house:
  - `AsbShng`: Asbestos Shingles
  - `AsphShn`: Asphalt Shingles
  - `BrkComm`: Brick Common
  - `BrkFace`: Brick Face
  - `CBlock`: Cinder Block
  - `CemntBd`: Cement Board
  - `HdBoard`: Hard Board
  - `ImStucc`: Imitation Stucco
  - `MetalSd`: Metal Siding
  - `Other`: Other
  - `Plywood`: Plywood
  - `PreCast`: PreCast
  - `Stone`: Stone
  - `Stucco`: Stucco
  - `VinylSd`: Vinyl Siding
  - `Wd Sdng`: Wood Siding
  - `WdShing`: Wood Shingles
- **Exterior2nd (Nominal)**: Exterior covering on house (if more than one material):
  - `AsbShng`: Asbestos Shingles
  - `AsphShn`: Asphalt Shingles
  - `BrkComm`: Brick Common
  - `BrkFace`: Brick Face
  - `CBlock`: Cinder Block
  - `CemntBd`: Cement Board
  - `HdBoard`: Hard Board
  - `ImStucc`: Imitation Stucco
  - `MetalSd`: Metal Siding
  - `Other`: Other
  - `Plywood`: Plywood
  - `PreCast`: PreCast
  - `Stone`: Stone
  - `Stucco`: Stucco
  - `VinylSd`: Vinyl Siding
  - `Wd Sdng`: Wood Siding
  - `WdShing`: Wood Shingles
- **MasVnrType (Nominal)**: Masonry veneer type:
  - `BrkCmn`: Brick Common
  - `BrkFace`: Brick Face
  - `CBlock`: Cinder Block
  - `Stone`: Stone
- **MasVnrArea (Continuous)**: Masonry veneer area in square feet (NA = missing data)
- **ExterQual (Ordinal)**: Evaluates the quality of the material on the exterior:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Average/Typical
  - `Fa`: Fair
  - `Po`: Poor
- **ExterCond (Ordinal)**: Present condition of the material on the exterior:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Average/Typical
  - `Fa`: Fair
  - `Po`: Poor

#### Foundation Details (Specific)
- **Foundation (Nominal)**: Type of foundation:
  - `BrkTil`: Brick & Tile
  - `CBlock`: Cinder Block
  - `PConc`: Poured Concrete
  - `Slab`: Slab
  - `Stone`: Stone
  - `Wood`: Wood

#### Basement Details (Specific)
- **BsmtQual (Ordinal)**: Height of the basement:
  - `Ex`: Excellent (100+ inches)
  - `Gd`: Good (90-99 inches)
  - `TA`: Typical (80-89 inches)
  - `Fa`: Fair (70-79 inches)
  - `Po`: Poor (<70 inches)
  - `NA`: No basement
- **BsmtCond (Ordinal)**: General condition of the basement:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Typical - slight dampness allowed
  - `Fa`: Fair - dampness or some cracking or settling
  - `Po`: Poor - Severe cracking, settling, or wetness
  - `NA`: No basement
- **BsmtExposure (Ordinal)**: Walkout or garden level basement walls:
  - `Gd`: Good Exposure
  - `Av`: Average Exposure (split levels or foyers typically score average or above)
  - `Mn`: Minimum Exposure
  - `No`: No Exposure
  - `NA`: No basement
- **BsmtFinType1 (Ordinal)**: Quality of basement finished area:
  - `GLQ`: Good Living Quarters
  - `ALQ`: Average Living Quarters
  - `BLQ`: Below Average Living Quarters
  - `Rec`: Average Rec Room
  - `LwQ`: Low Quality
  - `Unf`: Unfinished
  - `NA`: No basement
- **BsmtFinSF1 (Continuous)**: Type 1 finished square feet
- **BsmtFinType2 (Ordinal)**: Quality of second finished area (if present):
  - `GLQ`: Good Living Quarters
  - `ALQ`: Average Living Quarters
  - `BLQ`: Below Average Living Quarters
  - `Rec`: Average Rec Room
  - `LwQ`: Low Quality
  - `Unf`: Unfinished
  - `NA`: No basement
- **BsmtFinSF2 (Continuous)**: Type 2 finished square feet
- **BsmtUnfSF (Continuous)**: Unfinished square feet of basement area
- **TotalBsmtSF (Continuous)**: Total square feet of basement area

#### Heating and Cooling (Specific)
- **Heating (Nominal)**: Type of heating:
  - `Floor`: Floor Furnace
  - `GasA`: Gas forced warm air furnace
  - `GasW`: Gas hot water or steam heat
  - `Grav`: Gravity furnace
  - `OthW`: Hot water or steam heat other than gas
  - `Wall`: Wall furnace
- **HeatingQC (Ordinal)**: Heating quality and condition:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Average/Typical
  - `Fa`: Fair
  - `Po`: Poor
- **CentralAir (Nominal)**: Central air conditioning:
  - `N`: No
  - `Y`: Yes
- **Electrical (Ordinal)**: Electrical system:
  - `SBrkr`: Standard Circuit Breakers & Romex
  - `FuseA`: Fuse Box over 60 AMP and all Romex wiring (Average)
  - `FuseF`: 60 AMP Fuse Box and mostly Romex wiring (Fair)
  - `FuseP`: 60 AMP Fuse Box and mostly knob & tube wiring (poor)
  - `Mix`: Mixed
  - `NA`: Missing data

#### Floor Details (Specific)
- **1stFlrSF (Continuous)**: First Floor square feet
- **2ndFlrSF (Continuous)**: Second floor square feet
- **LowQualFinSF (Continuous)**: Low quality finished square feet (all floors)
- **GrLivArea (Continuous)**: Above grade (ground) living area square feet

#### Bathroom Details (Specific)
- **BsmtFullBath (Discrete)**: Basement full bathrooms
- **BsmtHalfBath (Discrete)**: Basement half bathrooms
- **FullBath (Discrete)**: Full bathrooms above grade
- **HalfBath (Discrete)**: Half baths above grade

#### Room Details (Specific)
- **BedroomAbvGr (Discrete)**: Number of bedrooms above grade (does NOT include basement bedrooms)
- **KitchenAbvGr (Discrete)**: Number of kitchens above grade
- **KitchenQual (Ordinal)**: Kitchen quality:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Typical/Average
  - `Fa`: Fair
  - `Po`: Poor
  - `NA`: Missing data
- **TotRmsAbvGrd (Discrete)**: Total rooms above grade (does not include bathrooms)

#### Functional Details (Aggregated)
- **Functional (Ordinal)**: Home functionality rating:
  - `Typ`: Typical Functionality
  - `Min1`: Minor Deductions 1
  - `Min2`: Minor Deductions 2
  - `Mod`: Moderate Deductions
  - `Maj1`: Major Deductions 1
  - `Maj2`: Major Deductions 2
  - `Sev`: Severely Damaged
  - `Sal`: Salvage only

#### Fireplace Details (Specific)
- **Fireplaces (Discrete)**: Number of fireplaces
- **FireplaceQu (Ordinal)**: Fireplace quality:
  - `Ex`: Excellent - Exceptional Masonry Fireplace
  - `Gd`: Good - Masonry Fireplace in main level
  - `TA`: Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
  - `Fa`: Fair - Prefabricated Fireplace in basement
  - `Po`: Poor - Ben Franklin Stove
  - `NA`: No fireplace

#### Garage Details (Specific)
- **GarageType (Nominal)**: Garage location:
  - `2Types`: More than one type of garage
  - `Attchd`: Attached to home
  - `Basment`: Basement Garage
  - `BuiltIn`: Built-In (Garage part of house - typically has room above garage)
  - `CarPort`: Car Port
  - `Detchd`: Detached from home
  - `No Garage`: No Garage
- **GarageFinish (Ordinal)**: Interior finish of the garage:
  - `Fin`: Finished
  - `RFn`: Rough Finished
  - `Unf`: Unfinished
  - `NA`: Missing
- **GarageCars (Discrete)**: Size of garage in car capacity
- **GarageArea (Continuous)**: Size of garage in square feet
- **GarageQual (Ordinal)**: Garage quality:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Typical/Average
  - `Fa`: Fair
  - `Po`: Poor
  - `NA`: Missing
- **GarageCond (Ordinal)**: Garage condition:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Typical/Average
  - `Fa`: Fair
  - `Po`: Poor
  - `NA`: Missing

#### Driveway Details (Specific)
- **PavedDrive (Ordinal)**: Paved driveway:
  - `Y`: Paved
  - `P`: Partial Pavement
  - `N`: Dirt/Gravel

#### Porch and Deck Details (Specific)
- **WoodDeckSF (Continuous)**: Wood deck area in square feet
- **OpenPorchSF (Continuous)**: Open porch area in square feet
- **EnclosedPorch (Continuous)**: Enclosed porch area in square feet
- **3SsnPorch (Continuous)**: Three season porch area in square feet
- **ScreenPorch (Continuous)**: Screen porch area in square feet

#### Pool Details (Specific)
- **PoolArea (Continuous)**: Pool area in square feet
- **PoolQC (Ordinal)**: Pool quality:
  - `Ex`: Excellent
  - `Gd`: Good
  - `TA`: Average/Typical
  - `Fa`: Fair
  - `NA`: No pool

#### Fence Details (Specific)
- **Fence (Ordinal)**: Fence quality:
  - `GdPrv`: Good Privacy
  - `MnPrv`: Minimum Privacy
  - `GdWo`: Good Wood
  - `MnWw`: Minimum Wood/Wire
  - `NA`: No fence

#### Miscellaneous Details (Specific)
- **MiscFeature (Nominal)**: Miscellaneous feature not covered in other categories:
  - `Elev`: Elevator
  - `Gar2`: 2nd Garage (if not described in garage section)
  - `Othr`: Other
  - `Shed`: Shed (over 100 SF)
  - `TenC`: Tennis Court
  - `NA`: None
- **MiscVal (Continuous)**: $Value of miscellaneous feature

#### Sale Timing (General)
- **MoSold (Discrete)**: Month Sold (MM)
- **YrSold (Discrete)**: Year Sold (YYYY)
