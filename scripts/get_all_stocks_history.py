# scripts/get_all_stocks_history.py
import sys
import os
import logging
from datetime import datetime
import pandas as pd

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.base import DataFetcherManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    manager = DataFetcherManager()
    
    # 1. 获取全市场股票列表
    logger.info("=" * 50)
    logger.info("正在获取全市场股票列表...")
    logger.info("=" * 50)
    
    try:
        stock_list = manager.get_all_stock_list()
        logger.info(f"获取成功，共 {len(stock_list)} 只股票")
        print("\n前5只股票:")
        print(stock_list.head())
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return

    # 2. 批量获取历史数据 (演示前5只)
    logger.info("\n" + "=" * 50)
    logger.info("开始演示批量获取历史数据 (前5只)...")
    logger.info("=" * 50)
    
    # 取前5只股票
    target_stocks = stock_list['code'].head(5).tolist()
    
    results = {}
    success_count = 0
    
    for code in target_stocks:
        try:
            # 获取最近30天数据
            df, source = manager.get_daily_data(code, days=30)
            results[code] = df
            success_count += 1
            logger.info(f"股票 {code} 获取成功 (来源: {source}), 数据量: {len(df)}")
        except Exception as e:
            logger.error(f"股票 {code} 获取失败: {e}")

    logger.info(f"\n批量获取演示完成: 成功 {success_count}/{len(target_stocks)}")

if __name__ == "__main__":
    main()
