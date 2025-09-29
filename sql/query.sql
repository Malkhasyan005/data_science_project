--Task 1A
-- Q1
select count(distinct store_id), count(distinct city_id)
from raw_sales_data;

-- Q2
select max(dt), min(dt)
from raw_sales_data;

-- Q3
select 
	count(distinct product_id), 
	count(distinct first_category_id),
	count(distinct second_category_id),
	count(distinct third_category_id)
from raw_sales_data;

-- Task 1B
-- Q4
select sum(sale_amount)
from raw_sales_data;

-- Q5
select sum(sale_amount) / (count(*) * 24)
from raw_sales_data;

-- Q6
select 
	sum(case
		when stock_hour6_22_cnt = 0 then 1
		else 0
	end) as "0 sales",
	sum(case
		when stock_hour6_22_cnt > 0 then 1
		else 0 
	end) as "non 0 sales"
from raw_sales_data;

-- Task 2A
-- Q7
SELECT
    hour_index AS hour_of_day,
    SUM(NULLIF(sale_value, '')::numeric) AS total_sales
FROM raw_sales_data
cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
ON true
GROUP BY hour_index
ORDER BY hour_index;


-- Q8
select 
	EXTRACT(dow from dt) as weekday,
	sum(sale_amount)
from raw_sales_data
group by weekday;


-- Task 2B
-- Q9
select 
	holiday_flag,
	avg(sale_amount) as avgsales
from raw_sales_data
group by holiday_flag;


-- Q10
select
	case
		when precpt > 0 then 1
		else 0
	end as is_rainy,
	avg(sale_amount)
from raw_sales_data
group by is_rainy;


-- Task 3A
create view hourly_business_summary_KM as
select
	city_id,
	store_id,
	product_id,
	holiday_flag,
	precpt,
	discount,
	dt,
	extract(month FROM dt) as "month",
	extract(dow from dt) as day_of_week,
	case when extract(dow from dt) in (0,6) then 1 else 0 end as is_weekend,
	case when (hour_index - 1) between 0 and 5 then 1 --night
		 when (hour_index - 1) between 6 and 11 then 2 --morning
		 when (hour_index - 1) between 12 and 15 then 3 --midday
		 when (hour_index - 1) between 16 and 19 then 4 --evening
		 when (hour_index - 1) between 20 and 23 then 5 --late
	end as part_of_day,
	case when discount > 0 then 1 else 0 end as promo_flag,
	case when precpt > 0 then 1 else 0 end as is_rainy,
	case when val::numeric > 0 then 1 else 0 end as activity_flag,
	case when ((hour_index - 1) between 6 and 22) and stock_hour6_22_cnt > 0 then 1 else 0 end as stockout_risk_hour,
	val::numeric AS sale_amount_hour,
    (hour_index - 1) AS hour_of_day
    FROM raw_sales_data,
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)


-- test
select city_id, sum(sale_amount_hour)
from hourly_business_summary_km
where is_weekend = 1 and promo_flag = 1
group by city_id;


-- Task 3B
create view store_performance_dashboard_km as
WITH unnested AS (
    SELECT
        rsd.store_id,
        rsd.sale_amount ,
        rsd.discount ,
        rsd.product_id,
        case when discount > 0 then 1 else 0 end as promo_flag,
        rsd.dt,
        val::numeric AS sale_value,
        (hour_index - 1) AS hour_of_day
    FROM raw_sales_data rsd
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
)
SELECT
    store_id,
    COUNT(DISTINCT product_id) AS num_products,
    COUNT(DISTINCT dt) AS num_days_tracked,
    COUNT(*) AS observed_hours,
    sum(sale_amount) as total_units,
    (sum(sale_amount ) / COUNT(*)) as units_per_hour,
    SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END) AS active_hours,
    (sum(sale_amount ) / SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END)) as units_per_active_hour,
    sum(case when sale_amount = 0 then 1 else 0 end) as  stockout_hours,
    (1 - sum(case when sale_amount = 0 then 1 else 0 end)::numeric / count(*)) as Stock_availability_rate,
    (SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END) / COUNT(*))::numeric as active_hour_ratio,
    (sum(sale_amount) / COUNT(DISTINCT dt)) as units_per_day,
	(sum(case when promo_flag = 1 then sale_amount else 0 end) / sum(sale_amount)) as promo_share,
	SUM(CASE WHEN promo_flag = 1 THEN sale_amount ELSE 0 END) AS promo_units,
	SUM(CASE WHEN promo_flag = 0 THEN sale_amount ELSE 0 END) AS non_promo_units,
	(stddev(sale_amount) / avg(sale_amount)) as sales_cv,
	avg(case when (extract(DOW from dt))::int in (0,6) then sale_amount else  0 end) as avg_weekend_units,
	avg(case when (extract(DOW from dt))::int between 1 and 5 then sale_amount else  0 end) as avg_weekday_units,
	(avg(case when (extract(DOW from dt))::int in (0,6) then sale_amount else  0 end)
	/ avg(case when (extract(DOW from dt))::int between 1 and 5 then sale_amount else  0 end)) as weekend_weekday_ratio
from unnested
GROUP BY store_id;


--Task 3C:
CREATE VIEW category_intelligence_ms AS
with unnested as (
	select *,
		case when discount > 0 then 1 else 0 end as promo_flag,
		val::numeric AS sale_amount_hour,
		    (hour_index - 1) AS hour_of_day  
		FROM raw_sales_data
		cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	    ) WITH ORDINALITY AS t(val, hour_index)
)
select 
	first_category_id,
	second_category_id ,
	third_category_id,
	promo_flag,
	(count(distinct store_id )) as num_stores,
	(count(distinct product_id)) as num_products,
	(sum(sale_amount )) as toal_units,
	SUM(CASE WHEN promo_flag = 1 THEN sale_amount ELSE 0 END) AS promo_units,
	SUM(CASE WHEN promo_flag = 0 THEN sale_amount ELSE 0 END) AS non_promo_units,
	(SUM(CASE WHEN promo_flag = 1 THEN sale_amount ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN promo_flag = 0 THEN sale_amount ELSE 0 END),0)) AS promo_lift_index,
	count(*)  as observed_hours,
	(sum(sale_amount ) / count(*)) as avg_units_per_hour,
	(stddev(sale_amount )) as sales_volatility,
	sum(case when sale_amount_hour = 0 then 1 else 0 end) AS stockout_hours,
	(sum(sale_amount ) / count(distinct store_id)) as units_per_store,
	(1 - sum(case when sale_amount_hour = 0 then 1 else 0 end)::numeric/count(*)) as stock_availability_rate,
COALESCE(
   (AVG(CASE WHEN sale_amount_hour > 0 THEN sale_amount_hour END) * SUM(CASE WHEN sale_amount_hour = 0 THEN 1 END)), 0) AS lost_units_due_to_stockouts,
	(avg(case when precpt > 0 then sale_amount_hour  end)) as avg_units_when_precip_is_more_than_zero,	
	(avg(case when holiday_flag = 1 then sale_amount else 0 end )) as avg_units_on_holidays,
	(avg(case when holiday_flag = 0 then sale_amount else 0 end )) as avg_units_non_holidays,
	(avg(case when holiday_flag = 1 then sale_amount else 0 end ) / avg(case when holiday_flag = 0 then sale_amount else 0 end )) as holiday_lift_index,
	(avg(CASE WHEN EXTRACT(DOW FROM dt) IN (0,6) THEN sale_amount_hour END) / avg(CASE WHEN EXTRACT(DOW FROM dt) not IN (0,6) THEN sale_amount_hour end)) as weekend_weekday_ratio,
	(stddev(case when holiday_flag = 1 then sale_amount_hour else 0 end) / stddev(case when holiday_flag = 0 then sale_amount_hour else 0 end )) as holiday_vs_nonholiday_variability
	
from unnested
group by first_category_id,second_category_id ,third_category_id,promo_flag;

--Test your view: Write a query that identifies categories with high promotion responsiveness but also high stockout challenges.
select 
        cim.first_category_id,
        cim.second_category_id,
        cim.third_category_id
from category_intelligence_ms cim 
where cim.stock_availability_rate < 0.2 and cim.holiday_lift_index > 1.2

--Task 3D:
CREATE VIEW agg AS
WITH time_series AS (
    SELECT 
        rsd.store_id,
        rsd.city_id,
        rsd.precpt,
        val::numeric AS sale_amount_hour,
        (hour_index - 1) AS hour_of_day,
        EXTRACT(DOW FROM dt)::int AS day_of_week,
        CASE WHEN EXTRACT(DOW FROM dt)::int IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
        EXTRACT(MONTH FROM dt)::int AS months,
        holiday_flag,
        CASE WHEN discount > 0 THEN 1 ELSE 0 END AS promo_flag
    FROM raw_sales_data rsd
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
)
SELECT
    hour_of_day,
    day_of_week,
    is_weekend,
    months,
    holiday_flag,
    promo_flag,
    sum(sale_amount_hour) AS total_units,
    avg(sale_amount_hour) AS avg_units_per_hour,
    COUNT(CASE when sale_amount_hour = 0 THEN 1 end) AS stockout_hours,
    (1 - count(CASE WHEN sale_amount_hour = 0 THEN 1 END)::numeric / count(*)) AS stock_availability_rate,
    COUNT(DISTINCT store_id) AS num_active_stores,
    SUM(sale_amount_hour)::numeric / NULLIF(COUNT(DISTINCT store_id),0) AS store_load_index,
    avg(precpt) AS avg_precip,
from time_series
GROUP BY hour_of_day, day_of_week, is_weekend, months, holiday_flag, promo_flag;



CREATE VIEW business_rhythm_patterns_ms AS
select *,
	case 
		when avg_units_per_hour > 1.2 * (select avg(avg_units_per_hour) from agg) then 'High Intensity'
		when avg_units_per_hour < 0.8 * (select avg(avg_units_per_hour) from agg) then 'Low Intensity'
		else 'Medium Intensity'
	end as performance_class
from agg;

--Test your view: Write a query that identifies the time periods requiring the highest operational intensity and support.
SELECT hour_of_day, day_of_week, is_weekend, holiday_flag, promo_flag,
       avg_units_per_hour, total_units, num_active_stores, store_load_index,
		performance_class
FROM business_rhythm_patterns_ms
WHERE performance_class = 'High Intensity'
ORDER BY avg_units_per_hour DESC;

--Task4A Question 11:
select (spdm.stockout_hours  / spdm.observed_hours ) as total_hours_experienced_stockouts
from store_performance_dashboard_ms spdm;

--Task4A Question 12:
select 
    'Hours_where_sales_more_than_zero' as Sales, 
    avg(hbsm.sale_amount_hour) as average
from hourly_business_summary_ms hbsm 
where hbsm.sale_amount_hour > 0 
union all
select 
    'Hours_where_sales_equal_to_zero' as Sales,
    avg(hbsm.sale_amount_hour) as average
from hourly_business_summary_ms hbsm 
where hbsm.sale_amount_hour = 0;

--Task4A Question 13:
with rankings as(
    select 
        hbsm.product_id as product,
        (sum(case when hbsm.sale_amount_hour = 0 then 1 else 0 end) * 1.0 / count(*)) as stockout_rate
    from hourly_business_summary_ms hbsm 
    group by product
)
select *,
	dense_rank() over(order by stockout_rate desc) as rnk
from rankings
limit 10;

--Task4B Question 14:
select 
    (sum(case when hbsm.sale_amount_hour = 0  and hbsm.activity_flag = 1 then 1 else 0 end) * 1.0 / count(*)) as stockout_rate_by_hour,
    hbsm.hour_of_day 
from hourly_business_summary_ms hbsm 
group by hbsm.hour_of_day; 

--Task4B Question 15:
select 
    spdm.store_id,
    spdm.stockout_hours,
    spdm.observed_hours,
    spdm.stock_availability_rate 
from store_performance_dashboard_ms spdm 
order by spdm.stockout_hours desc
limit 5;

--Task5A Question 16:
select 
    spdm.store_id,
    spdm.total_units 
from store_performance_dashboard_ms spdm 
order by spdm.total_units desc
limit 10;

--Task5A Question 17:
with tbl as (
    select 
        hbsm.city_id,
        hbsm.store_id,
        sum(hbsm.sale_amount_hour ) as sm
    from hourly_business_summary_ms hbsm 
    group by hbsm.city_id ,hbsm.store_id 
    )
select 
    city_id,
    avg(sm)
from tbl
group by city_id
order by avg(sm) desc;

--Task5A Question 18:
select 
    spdm.store_id,
    spdm.total_units,
    (spdm.stockout_hours * 1.0 / spdm.observed_hours ),
    spdm.stock_availability_rate 
from store_performance_dashboard_ms spdm 
group by spdm.store_id;

--Task5B Question 19:
select 
    cim.first_category_id,
    cim.second_category_id,
    cim.third_category_id,
    cim.toal_units 
from category_intelligence_ms cim 
order by cim.toal_units desc
limit 10;

--Task5B Question 20:
with tbl as (
    select 
        hbsm.product_id,
        avg(hbsm.sale_amount_hour ) as sales
    from hourly_business_summary_ms hbsm 
    group by hbsm.product_id
    )
select *,
	dense_rank() over(order by sales desc) as rnk
from tbl

--Task6A Question 21:
select 
    hbsm.promo_flag,
    avg(hbsm.sale_amount_hour)
from hourly_business_summary_ms hbsm 
group by hbsm.promo_flag 

--Task6A Question 22:
with discs as (
    select *,
        CASE
            WHEN discount = 1 THEN '0% (No discount)'
            WHEN discount < 0.99 AND discount > 0.90 THEN '1–10%'
            WHEN discount < 0.91 AND discount <= 0.80 THEN '11–20%'
            ELSE '21%+'
        end as ranges
    from hourly_business_summary_ms hbsm 
    )
select sum(sale_amount_hour) as sums,ranges
from discs
group by ranges

--Task6A Question 23:
SELECT 
    cim.first_category_id,
    cim.second_category_id,
    cim.third_category_id,
    cim.promo_units,
    cim.non_promo_units,
    cim.toal_units 
FROM category_intelligence_ms cim
ORDER BY cim.toal_units  DESC
LIMIT 10;

--Task6B Question 24:
select 
    hbsm.activity_flag,
    avg(hbsm.sale_amount_hour)
from hourly_business_summary_ms hbsm 
group by activity_flag 

--Task6B Question 25:
select hbsm.promo_flag ,avg(hbsm.sale_amount_hour)
from hourly_business_summary_ms hbsm 
group by hbsm.promo_flag 

--Task7A Question 26:
WITH unnested AS (
    select *,
        val::numeric AS sale_amount_hour,
        (hour_index - 1) AS hour_of_day
    FROM raw_sales_data rsd
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
)
select 
    store_id,product_id,
    avg(case when sale_amount_hour > 0 then sale_amount_hour end) as avg_in_stock_sales,
    (sum(case when sale_amount_hour = 0 then 1 else 0 end) / count(*)) as stockout_rate
from unnested
group by store_id,product_id
order by avg_in_stock_sales desc
limit 5

--Task7A Question 27:
WITH unnested AS (
    select *,
        val::numeric AS sale_amount_hour,
        (hour_index - 1) AS hour_of_day
    FROM raw_sales_data rsd
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
),
tble as(
    select *,
        case
            when avg_temperature < 10 then 'Cold'
            when avg_temperature >= 10 and avg_temperature < 20 then 'Mild'
            when avg_temperature >= 20 and avg_temperature < 30 then 'Warm'
            else 'Hot'
        end as temperature_range
    from unnested 
)
select 
    temperature_range,
    first_category_id,
    second_category_id,
    third_category_id,
    sum(sale_amount_hour) as total
from tble
group by temperature_range,first_category_id,second_category_id,third_category_id
order by temperature_range

--Task7B Question 28:
WITH unnested AS (
    select *,
        val::numeric AS sale_amount_hour,
        (hour_index - 1) AS hour_of_day
    FROM raw_sales_data rsd
    cross join lateral unnest(
		string_to_array(
			REPLACE(REPLACE(REPLACE(hours_stock_status, '[', ''), ']', ''), ' ', ','), ',')
	) WITH ORDINALITY AS t(val, hour_index)
)
select 
    product_id,
    count(*) as total_hours,
    (sum(case when sale_amount_hour = 0 then 1 else 0 end)) as stockout,
    (sum(case when sale_amount_hour = 0 then 1 else 0 end) / count(*)) as stockout_rate
from unnested
where hour_of_day in (8,9,10)
group by product_id

--Task7B Question 29:
select cim.lost_units_due_to_stockouts 
from category_intelligence_ms cim 

----Task7B Question 30:
--import pandas as pd
--import matplotlib.pyplot as plt
--from sklearn.preprocessing import StandardScaler
--from sklearn.decomposition import PCA
--from sklearn.cluster import KMeans
--from sklearn.metrics import silhouette_score
--
--# === Step 1: Load your CSV ===
--df = pd.read_csv("/home/rog/Documents/Titanic/store_performance_dashboard_ms_202509281824.csv")
--
--# Keep only numeric features (drop IDs)
--features = df.drop(columns=["store_id"])  
--
--# === Step 2: Scale the features ===
--scaler = StandardScaler()
--scaled = scaler.fit_transform(features)
--
--# === Step 3: PCA for dimensionality reduction ===
--pca = PCA(n_components=0.9)   # keep 90% variance
--pca_data = pca.fit_transform(scaled)
--print("Explained variance ratio (per component):", pca.explained_variance_ratio_)
--
--# === Step 4: KMeans++ with different k values ===
--inertias = []
--silhouettes = []
--K = range(2, 11)
--
--for k in K:
--    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
--    labels = kmeans.fit_predict(pca_data)
--    inertias.append(kmeans.inertia_)
--    silhouettes.append(silhouette_score(pca_data, labels))
--
--# === Step 5: Plot elbow method ===
--plt.figure(figsize=(12,5))
--
--plt.subplot(1,2,1)
--plt.plot(K, inertias, 'bo-')
--plt.xlabel("k")
--plt.ylabel("Inertia")
--plt.title("Elbow Method")
--
--plt.subplot(1,2,2)
--plt.plot(K, silhouettes, 'ro-')
--plt.xlabel("k")
--plt.ylabel("Silhouette Score")
--plt.title("Silhouette Score vs k")
--
--plt.show()
--
--# === Step 6: Choose best k (from plots) ===
--best_k = 4  # <-- manually decide after seeing elbow/silhouette
--
--kmeans = KMeans(n_clusters=best_k, init="k-means++", random_state=42)
--df["health_cluster"] = kmeans.fit_predict(pca_data)
--
--# === Step 7: Optional continuous health score ===
--# Distance to cluster center (smaller distance = "healthier")
--distances = kmeans.transform(pca_data)
--df["health_score"] = -distances.min(axis=1)  # negate so higher = healthier
--
--# === Normalize health_score to 0–100 ===
--min_val = df["health_score"].min()
--max_val = df["health_score"].max()
--
--df["health_index"] = 100 * (df["health_score"] - min_val) / (max_val - min_val)
--
--# === Map cluster names ===
--cluster_names = {
--    0: "Struggling Stores",
--    1: "Stable but Low Performers",
--    2: "Mid-Tier Performers",
--    3: "High Performers"
--}
--
--df["cluster_name"] = df["health_cluster"].map(cluster_names)
--
--# Save results
--df.to_csv("store_health_scores_named.csv", index=False)
--print(df[["store_id", "cluster_name", "health_index"]].head())




