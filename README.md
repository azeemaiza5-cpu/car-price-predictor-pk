Pakistan Used Car Price Predictor

A model that estimates what a used car should sell for in Pakistan, based on things like brand, age, mileage, and engine size. I picked this problem myself, cleaned the raw data, tried a few models, and deployed the winner as a live app instead of leaving it sitting in a notebook.

Live app: https://car-price-predictor-pk-g98ysl4p3sctq4jfjftcbd.streamlit.app/

Why this problem

Car pricing here is pretty informal — you ask around, check a few PakWheels listings, maybe haggle based on vibes. I wanted to see if a model trained on actual listing data could give a more grounded starting point than "ask your cousin who knows about cars." It's also a dataset I could sanity-check myself, since I roughly know what a 2018 Corolla or a Wagon R should go for.

The data

Used a PakWheels listings dataset from Kaggle — about 60k rows to start, down to ~57k after cleaning. It came with the usual scraped-data mess: prices and mileage stored as text like "26,755 km" instead of numbers, a column literally named "Model" that actually held the year, brand buried inside a messy title string, and a few cars supposedly manufactured in 1940. Cleaning this up was honestly most of the actual work.

What I did
Pulled the brand out of the title text
Fixed the mislabeled year column and cleaned CC/mileage from text into numbers
Dropped rows with missing prices and unrealistic years (kept 1990–2024)
Looked at the price distribution and price vs. age before modeling — car pricing turned out to be pretty non-linear, which mattered for model choice later
Grouped the rarest brands (out of 58 total) into "Other" since some had only a handful of listings — didn't want the model overfitting on a brand it saw three times
Engineered a Car_Age feature instead of using raw year
One-hot encoded brand, engine type, and transmission
Models I tried
Model	MAE (Lacs)	R²
Linear Regression	11.12	0.41
Random Forest	~4.0	~0.85
XGBoost	4.50	0.84

Linear Regression was clearly the wrong tool — price doesn't move in a straight line with age or mileage, it's more like factors compounding on each other (an old car with high mileage is worse off than either problem alone). Random Forest picked up on that and won, just ahead of XGBoost.

For deployment, I actually used a smaller Random Forest than my best-performing one (fewer trees, capped depth) — the full-size model file was too large to upload cleanly, so I traded a small amount of accuracy for something that would actually deploy and run. Worth it.

Honest limitations
This is trained on a snapshot of listings, not live market data, so it won't catch sudden price shifts from import duty changes or exchange rate swings
Rare brands and very old or very new cars have less data behind them, so those predictions are less reliable
It's meant as a starting estimate, not something to base a real negotiation entirely on
Running it yourself
bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
Built with

Python, pandas, scikit-learn, Streamlit
