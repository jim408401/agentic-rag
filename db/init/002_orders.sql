-- 模擬的客戶訂單業務資料庫（Phase 3 的 Text-to-SQL 工具會查這裡）
-- 放在獨立 schema，之後用唯讀角色限制 Agent 只能讀這個 schema。
CREATE SCHEMA IF NOT EXISTS sales;

CREATE TABLE sales.customers (
    customer_id   SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    tier          TEXT NOT NULL CHECK (tier IN ('一般', '金卡', '白金')),
    region        TEXT NOT NULL,
    created_at    DATE NOT NULL
);
COMMENT ON TABLE sales.customers IS '客戶主檔；tier 為會員等級，影響退貨期限與運費';

CREATE TABLE sales.products (
    product_id    SERIAL PRIMARY KEY,
    sku           TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    NUMERIC(10, 2) NOT NULL
);
COMMENT ON TABLE sales.products IS '商品主檔；unit_price 為未稅單價（新台幣）';

CREATE TABLE sales.orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   INT NOT NULL REFERENCES sales.customers,
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('處理中', '已出貨', '已送達', '已取消', '退貨中', '已退貨')),
    shipped_at    DATE,
    delivered_at  DATE
);
COMMENT ON TABLE sales.orders IS '訂單主檔；營收只計算 status 為 已出貨、已送達 的訂單';

CREATE TABLE sales.order_items (
    order_id      TEXT NOT NULL REFERENCES sales.orders,
    product_id    INT NOT NULL REFERENCES sales.products,
    quantity      INT NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
COMMENT ON TABLE sales.order_items IS '訂單明細；unit_price 為成交當下單價';

INSERT INTO sales.customers (name, tier, region, created_at) VALUES
    ('宏遠科技股份有限公司', '白金', '台北', '2021-03-15'),
    ('綠野餐飲有限公司', '一般', '台中', '2022-07-01'),
    ('海晴貿易', '金卡', '高雄', '2020-11-20'),
    ('青石設計工作室', '一般', '台北', '2024-02-10'),
    ('北辰物流股份有限公司', '金卡', '桃園', '2019-05-05'),
    ('晨曦診所', '一般', '新竹', '2023-09-18');

INSERT INTO sales.products (sku, name, category, unit_price) VALUES
    ('NB-1401', '商務筆電 14 吋', '電腦', 32900),
    ('MN-2701', '27 吋 4K 螢幕', '週邊', 11500),
    ('KB-0102', '無線鍵盤滑鼠組', '週邊', 1290),
    ('DK-3300', 'USB-C 擴充基座', '週邊', 4590),
    ('PR-5050', '雷射多功能事務機', '辦公設備', 15800),
    ('SW-0001', '防毒軟體一年授權', '軟體', 990);

INSERT INTO sales.orders (order_id, customer_id, order_date, status, shipped_at, delivered_at) VALUES
    ('SO-2026-0001', 1, '2026-07-02', '已送達', '2026-07-03', '2026-07-05'),
    ('SO-2026-0002', 2, '2026-07-10', '已送達', '2026-07-11', '2026-07-14'),
    ('SO-2026-0003', 3, '2026-07-21', '已退貨', '2026-07-22', '2026-07-24'),
    ('SO-2026-0004', 1, '2026-08-04', '已送達', '2026-08-05', '2026-08-06'),
    ('SO-2026-0005', 4, '2026-08-12', '已取消', NULL, NULL),
    ('SO-2026-0006', 5, '2026-08-19', '已送達', '2026-08-20', '2026-08-21'),
    ('SO-2026-0007', 6, '2026-09-01', '已出貨', '2026-09-02', NULL),
    ('SO-2026-0008', 3, '2026-09-09', '退貨中', '2026-09-10', '2026-09-12'),
    ('SO-2026-0009', 5, '2026-09-23', '已送達', '2026-09-24', '2026-09-26'),
    ('SO-2026-0010', 2, '2026-10-01', '處理中', NULL, NULL),
    ('SO-2026-0011', 1, '2026-10-05', '已出貨', '2026-10-06', NULL),
    ('SO-2026-0012', 4, '2026-10-08', '處理中', NULL, NULL);

INSERT INTO sales.order_items (order_id, product_id, quantity, unit_price) VALUES
    ('SO-2026-0001', 1, 10, 31900), ('SO-2026-0001', 2, 10, 11000),
    ('SO-2026-0002', 3, 4, 1290),
    ('SO-2026-0003', 5, 1, 15800),
    ('SO-2026-0004', 4, 10, 4390), ('SO-2026-0004', 6, 20, 900),
    ('SO-2026-0005', 1, 1, 32900),
    ('SO-2026-0006', 1, 5, 32400), ('SO-2026-0006', 3, 5, 1290),
    ('SO-2026-0007', 5, 1, 15800), ('SO-2026-0007', 6, 3, 990),
    ('SO-2026-0008', 2, 2, 11500),
    ('SO-2026-0009', 4, 8, 4590), ('SO-2026-0009', 2, 4, 11200),
    ('SO-2026-0010', 3, 6, 1290),
    ('SO-2026-0011', 1, 3, 31900),
    ('SO-2026-0012', 6, 5, 990);
