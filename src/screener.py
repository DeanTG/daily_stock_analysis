# -*- coding: utf-8 -*-
"""
===================================
A股智能选股器
===================================

职责：
1. 从全市场筛选符合特定条件的股票
2. 实现多维度过滤（价格、市值、财务、技术形态）
3. 专注于“主力吸筹、涨停未大涨”等特定模式

选股逻辑：
1. 基础过滤：
   - 剔除价格 > 20元
   - 剔除大市值 (> 200亿)
   - 剔除亏损股 (PE < 0)
2. 形态过滤 (需获取日线数据)：
   - 近3个月(60天)有涨停记录 (涨幅 > 9.5%)
   - 近3个月涨幅不大 (区间涨幅 < 30%)
   - 均线多头排列 (MA5 > MA10 > MA20)
   - 处于相对低位 (当前价 < 近250日最高价的 50% 或 处于近250日分位数 < 30%)
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_provider.base import DataFetcherManager
from src.stock_analyzer import StockTrendAnalyzer

logger = logging.getLogger(__name__)

class StockScreener:
    def __init__(self, max_workers: int = 5):
        self.fetcher_manager = DataFetcherManager()
        self.trend_analyzer = StockTrendAnalyzer()
        self.max_workers = max_workers

    def get_market_snapshot(self) -> pd.DataFrame:
        """
        获取全市场快照（包含基础指标：价格、市值、PE等）
        由于 efinance/akshare 的 get_all_stock_list 只返回代码和名称，
        我们需要尝试获取更详细的实时行情数据来做初筛。
        
        策略：
        使用 efinance.stock.get_realtime_quotes() 获取全市场实时行情，
        它包含了 价格、涨跌幅、成交量 等信息。
        但可能不包含 总市值、PE。
        
        如果缺少 PE/市值，可能需要额外获取或估算。
        efinance 的 get_realtime_quotes 返回列包含：
        股票代码, 股票名称, 涨跌幅, 最新价, 最高, 最低, 开盘, 成交量, 成交额, 换手率, 量比, 委比, 市盈率(动), 市净率, 总市值, 流通市值...
        (具体列名需确认，efinance 通常返回中文列名)
        """
        try:
            import efinance as ef
            import akshare as ak
            
            # 优先使用 akshare 获取（更稳定）
            try:
                logger.info("尝试使用 AkShare 获取全市场实时行情...")
                df = ak.stock_zh_a_spot_em()
                
                # 标准化列名 (Akshare -> 标准)
                rename_map = {
                    '代码': 'code',
                    '名称': 'name',
                    '最新价': 'price',
                    '市盈率-动态': 'pe',
                    '总市值': 'total_mv',
                    '流通市值': 'circ_mv',
                    '涨跌幅': 'pct_chg'
                }
                df = df.rename(columns=rename_map)
                logger.info(f"AkShare 获取成功，共 {len(df)} 条数据")
            except Exception as e:
                logger.warning(f"AkShare 获取失败 ({e})，切换到 efinance...")
                # 回退到 efinance
                logger.info("尝试使用 efinance 获取全市场实时行情快照...")
                df = ef.stock.get_realtime_quotes()
            
            # 打印列名以便调试
            # logger.debug(f"全市场数据列名: {df.columns.tolist()}")
            
            # 如果是 efinance 的列名，进行映射
            if '股票代码' in df.columns:
                rename_map = {
                    '股票代码': 'code',
                    '股票名称': 'name',
                    '最新价': 'price',
                    '市盈率(动)': 'pe',
                    '总市值': 'total_mv',
                    '流通市值': 'circ_mv',
                    '涨跌幅': 'pct_chg'
                }
                df = df.rename(columns=rename_map)
            
            # 确保数值列类型正确
            numeric_cols = ['price', 'pe', 'total_mv', 'circ_mv', 'pct_chg']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"获取全市场快照失败: {e}")
            return pd.DataFrame()

    def filter_basics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础条件过滤
        1. 价格 <= 20
        2. 总市值 <= 200亿 (200 * 10000 * 10000) -> efinance 返回单位通常是元? 需确认。
           efinance 总市值列通常是数值（元）。
        3. PE > 0 (剔除亏损)
        """
        if df.empty:
            return df
            
        logger.info(f"基础过滤前数量: {len(df)}")
        
        # 1. 价格过滤
        df = df[df['price'] <= 20]
        
        # 2. 亏损过滤 (PE > 0)
        # 注意：有些数据源 PE 为 "-" 或 NaN
        df = df[df['pe'] > 0]
        
        # 3. 市值过滤
        # efinance 的总市值单位通常是 元。200亿 = 200 * 10^8
        # 有些可能是 '-'
        # 先检查单位，通常 efinance 返回的是完整数值
        # 假设单位是元
        market_cap_limit = 200 * 100000000 
        df = df[df['total_mv'] <= market_cap_limit]
        
        logger.info(f"基础过滤后数量: {len(df)}")
        return df

    def check_technical_pattern(self, code: str, name: str) -> Optional[Dict]:
        """
        检查单只股票的技术形态（需获取历史数据）
        
        条件：
        1. 近3个月(60天)有涨停 (单日涨幅 > 9.5%)
        2. 近3个月涨幅不大 (区间涨幅 < 30%? 或者 50%?)
           这里定义为：(当前价 - 60天前价) / 60天前价 < 0.5
        3. 均线向上 (MA5 > MA10 > MA20)
        4. 月线低位 (这里简化为：当前价 < 近250日最高价 * 0.6，即回撤了40%以上，或者处于低位区间)
        """
        try:
            # 获取日线数据 (近 250 天，用于计算年线高点)
            df, _ = self.fetcher_manager.get_daily_data(code, days=250)
            
            if df is None or df.empty or len(df) < 60:
                return None
            
            # 按日期升序
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            
            # 取最近 60 天数据
            df_60 = df.tail(60)
            current_price = df.iloc[-1]['close']
            price_60_days_ago = df_60.iloc[0]['close']
            
            # 1. 检查是否有涨停 (涨幅 > 9.5%)
            # 科创板/创业板是 20%，主板是 10%。简单起见 > 9.5% 视为大阳线/涨停
            limit_up_days = df_60[df_60['pct_chg'] > 9.5]
            if len(limit_up_days) == 0:
                return None
            
            # 2. 检查区间涨幅 (滞涨)
            # 既然有涨停，但整体涨幅又不能太大（说明主力在吸筹，不是主升浪）
            interval_increase = (current_price - price_60_days_ago) / price_60_days_ago
            if interval_increase > 0.5: # 涨幅超过 50% 算大涨了，剔除
                return None
            
            # 3. 均线多头排列
            # 计算 MA
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma10 = df['close'].rolling(10).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            
            if not (ma5 > ma10 > ma20):
                return None
            
            # 4. 长期低位 (月线低位)
            # 使用近 250 天(约一年)的数据判断
            year_high = df['high'].max()
            year_low = df['low'].min()
            
            # 分位数：(当前 - 最低) / (最高 - 最低)
            # < 0.3 表示处于底部 30% 区间
            # 或者 < 最高价 * 0.6
            position_rank = (current_price - year_low) / (year_high - year_low) if year_high != year_low else 0
            
            # 宽松一点：0.5 (处于历史 50% 分位以下)
            if position_rank > 0.5: 
                return None
                
            return {
                'code': code,
                'name': name,
                'price': current_price,
                'limit_up_count': len(limit_up_days),
                'interval_increase': interval_increase * 100,
                'position_rank': position_rank * 100
            }
            
        except Exception as e:
            # logger.warning(f"分析 {code} 形态失败: {e}")
            return None

    def run_screen(self) -> List[Dict]:
        """
        执行选股流程
        """
        logger.info("开始执行智能选股...")
        
        # 1. 获取全市场快照
        df_market = self.get_market_snapshot()
        if df_market.empty:
            logger.error("无法获取市场快照，选股终止")
            return []
            
        # 2. 基础过滤
        df_basics = self.filter_basics(df_market)
        candidates = df_basics[['code', 'name']].to_dict('records')
        
        logger.info(f"进入形态分析阶段，候选股票数: {len(candidates)}")
        logger.info("正在批量获取历史数据进行形态匹配 (这可能需要几分钟)...")
        
        results = []
        # 使用线程池并发处理
        # 限制 max_workers 防止被封
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stock = {
                executor.submit(self.check_technical_pattern, stock['code'], stock['name']): stock 
                for stock in candidates
            }
            
            processed = 0
            total = len(candidates)
            
            for future in as_completed(future_to_stock):
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"进度: {processed}/{total}...")
                    
                res = future.result()
                if res:
                    results.append(res)
        
        logger.info(f"选股完成! 命中数量: {len(results)}")
        return results

if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    screener = StockScreener(max_workers=5)
    results = screener.run_screen()
    print("\n选股结果:")
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.sort_values('limit_up_count', ascending=False))
