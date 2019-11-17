# Python_project

Komandų vykdymo seka bei parametrų reikšmių paaiškinimai:
1. churn_sampling.py - failas skirtas sukurti (sumažinti) imtį iš dviejų failų: customer_usage_00004.csv ir customer_churn_00004.csv. Rezultas du csv failai: customer_usage.csv ir customer_churn.csv;
2. visualize.py - atsakingas už žvalgomąją analitiką ir yra skirtas apvalyti duomenis nuo kintamųjų, kurie koreliuoja. Kodas naudoja failą customer_usage.csv ir jį apdorojus gaunami: summary_usage.csv, corr_pearson_usage.csv, corr_spearman_usage.csv bei failai folderiuose "hist_csv"(duomenų failai) ir "hist_png"(grafikų nuotraukos);
3. BuildFeature.py - naudojamas duomenų apjungimui į vieną failą ir duomenų agragavimui. Naudojama data - customer_usage.csv, customer_churn.csv ir duomenys yra išsaugomi parquet formatu, išvestis: folderis - "sample_aggregated_usage_with_churn";
4. BuildClusteringFeatureRandomForestClassifier.py - skaito duomenis, kurie yra išsaugoti parquet formatu ir yra skirtas reikšmingų kitamųjų parinkimui pasinaudojant atsitiktinių miškų klasifikatoriumi; Išvestis - "failopavadinimas.json", čia yra rasti naudingi kintamieji;
5. Train_Clustering_model.py - failas, kurio kodas skirtas klasterių skaičiaus parinkimui pasinaudojant "k-means" algoritmą. Naudoja duomenis iš folderio "sample_aggregated_usage_with_churn". Gaunami metrikų grafikai: "k_fk.png" bei "k_sse.png" iš kurių galime nustatyti klasterių skaičių;
6. cluster2.py - imami duomenys parquet formatu iš folderio "sample_aggregated_usage_with_churn", tik jau su ekspertų pasirinktu klasterių skaičiumi. Gaunami denormalizuoti klasterių centrai ir parquet formatu išsaugotą "k-means" klasterizavimo modelį (folderis - "clustered_kmeans__k_5_parquet"). Šių gautų duomenų pagalba atliekamas duomenų segmentavimas (išskirstymas į klasterius).

Task'ai sau:
1. Sukoduoti taip, kad imtų reikiamą kelią iš atskiro failo, esančio projekto kataloge, kad kodas veiktų jame nekeičiant kelio į failus (to atskiro failo turinys skirtingas kiekvieno kompiutery, bet pavadinimas ir lokacija vienoda ir jo negalima pushinti į GIT);
2. b
