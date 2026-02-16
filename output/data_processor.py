"""
Data Processor Module
Handles CSV data loading, generation, and preprocessing for ML tasks.
Uses only Python standard library (csv module).
"""

import csv
import random
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class DataProcessor:
    """Process and manipulate CSV data for machine learning tasks."""
    
    def __init__(self):
        self.data = []
        self.headers = []
    
    def load_csv(self, filepath: str) -> bool:
        """Load data from a CSV file."""
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.headers = next(reader, [])
                self.data = [row for row in reader]
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def save_csv(self, filepath: str) -> bool:
        """Save current data to a CSV file."""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if self.headers:
                    writer.writerow(self.headers)
                writer.writerows(self.data)
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def generate_linear_data(self, num_samples: int = 100, 
                            noise: float = 0.1,
                            theta_0: float = 2.0,
                            theta_1: float = 3.0) -> List[List[float]]:
        """Generate synthetic linear data: y = theta_0 + theta_1 * x + noise."""
        self.headers = ['x', 'y']
        self.data = []
        
        for _ in range(num_samples):
            x = random.uniform(-10, 10)
            noise_val = random.gauss(0, noise)
            y = theta_0 + theta_1 * x + noise_val
            self.data.append([round(x, 4), round(y, 4)])
        
        return self.data
    
    def generate_classification_data(self, num_samples: int = 100,
                                    num_classes: int = 2,
                                    noise: float = 0.3) -> List[List[float]]:
        """Generate synthetic classification data."""
        self.headers = ['x1', 'x2', 'label']
        self.data = []
        
        for i in range(num_samples):
            # Generate points in a spiral pattern for non-linear classification
            angle = (i / num_samples) * 2 * math.pi * num_classes
            radius = random.uniform(0.5, 2.0)
            
            x1 = radius * math.cos(angle) + random.gauss(0, noise)
            x2 = radius * math.sin(angle) + random.gauss(0, noise)
            label = i % num_classes
            
            self.data.append([round(x1, 4), round(x2, 4), label])
        
        return self.data
    
    def generate_customer_data(self, num_samples: int = 50) -> List[List]:
        """Generate realistic-looking customer data with 50+ rows."""
        self.headers = ['customer_id', 'age', 'income', 'spending_score', 'loyalty_years', 'purchase_amount']
        self.data = []
        
        first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        for i in range(num_samples):
            customer_id = f"CUST{1000 + i}"
            age = random.randint(18, 75)
            income = random.randint(20000, 150000)
            spending_score = random.randint(1, 100)
            loyalty_years = random.randint(0, 20)
            purchase_amount = round(random.uniform(50.0, 5000.0), 2)
            
            self.data.append([customer_id, age, income, spending_score, loyalty_years, purchase_amount])
        
        return self.data
    
    def generate_sales_data(self, num_samples: int = 60) -> List[List]:
        """Generate realistic-looking sales data with 60+ rows."""
        self.headers = ['date', 'product_id', 'quantity', 'unit_price', 'total', 'region']
        self.data = []
        
        products = ['PROD001', 'PROD002', 'PROD003', 'PROD004', 'PROD005']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        start_year = 2023
        start_month = 1
        
        for i in range(num_samples):
            # Generate dates spread across multiple months
            month = start_month + (i % 12)
            day = (i % 28) + 1
            date = f"{start_year}-{month:02d}-{day:02d}"
            
            product_id = random.choice(products)
            quantity = random.randint(1, 100)
            unit_price = round(random.uniform(10.0, 500.0), 2)
            total = round(quantity * unit_price, 2)
            region = random.choice(regions)
            
            self.data.append([date, product_id, quantity, unit_price, total, region])
        
        return self.data
    
    def generate_employee_data(self, num_samples: int = 100) -> List[List]:
        """Generate realistic-looking employee data with salary and department info."""
        self.headers = ['employee_id', 'name', 'age', 'department', 'salary', 'years_experience']
        self.data = []
        
        first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Sarah']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Wilson', 'Anderson']
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'IT', 'Customer Support']
        
        for i in range(num_samples):
            employee_id = f"EMP{1000 + i}"
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            age = random.randint(22, 65)
            department = random.choice(departments)
            years_exp = random.randint(0, 40)
            
            # Salary based on department with some variation
            base_salaries = {
                'Engineering': 80000,
                'IT': 75000,
                'Finance': 70000,
                'Marketing': 65000,
                'Sales': 60000,
                'HR': 55000,
                'Operations': 50000,
                'Customer Support': 45000
            }
            base = base_salaries.get(department, 50000)
            salary = base + random.randint(-10000, 20000) + (years_exp * 1000)
            salary = max(30000, min(salary, 200000))  # Clamp between 30k and 200k
            
            self.data.append([employee_id, name, age, department, salary, years_exp])
        
        return self.data
    
    def normalize(self, columns: List[int] = None) -> Dict[str, Tuple[float, float]]:
        """Normalize data to [0, 1] range. Returns min/max values for each column."""
        if not self.data:
            return {}
        
        num_cols = len(self.data[0])
        if columns is None:
            columns = list(range(num_cols))
        
        # Find min and max for each column
        min_vals = [float('inf')] * num_cols
        max_vals = [float('-inf')] * num_cols
        
        for row in self.data:
            for col in columns:
                if col < len(row):
                    val = float(row[col])
                    min_vals[col] = min(min_vals[col], val)
                    max_vals[col] = max(max_vals[col], val)
        
        # Normalize the data
        normalized_data = []
        for row in self.data:
            new_row = list(row)
            for col in columns:
                if col < len(row):
                    val = float(row[col])
                    if max_vals[col] != min_vals[col]:
                        new_row[col] = round((val - min_vals[col]) / (max_vals[col] - min_vals[col]), 4)
                    else:
                        new_row[col] = 0.0
            normalized_data.append(new_row)
        
        self.data = normalized_data
        
        return {i: (min_vals[i], max_vals[i]) for i in columns}
    
    def get_features_targets(self, feature_cols: List[int], target_col: int) -> Tuple[List[List[float]], List[float]]:
        """Split data into features and targets."""
        features = []
        targets = []
        
        for row in self.data:
            feature_row = [float(row[col]) for col in feature_cols if col < len(row)]
            target_val = float(row[target_col]) if target_col < len(row) else 0
            features.append(feature_row)
            targets.append(target_val)
        
        return features, targets
    
    def shuffle_data(self):
        """Randomly shuffle the data rows."""
        random.shuffle(self.data)
    
    def split_data(self, train_ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]]]:
        """Split data into training and testing sets."""
        if not self.data:
            return [], []
        
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        return train_data, test_data
    
    # Statistical Analysis Methods
    
    def _get_column_values(self, column_name: str) -> List[float]:
        """Extract numeric values from a column by name."""
        if column_name not in self.headers:
            raise ValueError(f"Column '{column_name}' not found in headers: {self.headers}")
        
        col_idx = self.headers.index(column_name)
        values = []
        for row in self.data:
            if col_idx < len(row):
                try:
                    values.append(float(row[col_idx]))
                except (ValueError, TypeError):
                    continue
        return values
    
    def mean(self, column_name: str) -> float:
        """Calculate the mean of a numeric column."""
        values = self._get_column_values(column_name)
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def median(self, column_name: str) -> float:
        """Calculate the median of a numeric column."""
        values = self._get_column_values(column_name)
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        else:
            return sorted_values[mid]
    
    def std(self, column_name: str) -> float:
        """Calculate the standard deviation of a numeric column."""
        values = self._get_column_values(column_name)
        if len(values) < 2:
            return 0.0
        
        mean_val = self.mean(column_name)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def min_value(self, column_name: str) -> float:
        """Get the minimum value in a numeric column."""
        values = self._get_column_values(column_name)
        return min(values) if values else 0.0
    
    def max_value(self, column_name: str) -> float:
        """Get the maximum value in a numeric column."""
        values = self._get_column_values(column_name)
        return max(values) if values else 0.0
    
    def avg_salary_per_department(self, salary_col: str = 'salary', dept_col: str = 'department') -> Dict[str, float]:
        """Calculate average salary per department."""
        if salary_col not in self.headers or dept_col not in self.headers:
            raise ValueError(f"Required columns not found. Headers: {self.headers}")
        
        dept_salaries = defaultdict(list)
        salary_idx = self.headers.index(salary_col)
        dept_idx = self.headers.index(dept_col)
        
        for row in self.data:
            if salary_idx < len(row) and dept_idx < len(row):
                try:
                    salary = float(row[salary_idx])
                    dept = row[dept_idx]
                    dept_salaries[dept].append(salary)
                except (ValueError, TypeError):
                    continue
        
        return {dept: sum(salaries) / len(salaries) for dept, salaries in dept_salaries.items()}
    
    def employee_count_per_department(self, dept_col: str = 'department') -> Dict[str, int]:
        """Count employees per department."""
        if dept_col not in self.headers:
            raise ValueError(f"Department column '{dept_col}' not found in headers: {self.headers}")
        
        dept_counts = defaultdict(int)
        dept_idx = self.headers.index(dept_col)
        
        for row in self.data:
            if dept_idx < len(row):
                dept = row[dept_idx]
                dept_counts[dept] += 1
        
        return dict(dept_counts)
    
    def age_distribution(self, age_col: str = 'age', bins: List[Tuple[int, int]] = None) -> Dict[str, int]:
        """Get age distribution across specified bins."""
        if age_col not in self.headers:
            raise ValueError(f"Age column '{age_col}' not found in headers: {self.headers}")
        
        age_idx = self.headers.index(age_col)
        ages = []
        
        for row in self.data:
            if age_idx < len(row):
                try:
                    ages.append(int(row[age_idx]))
                except (ValueError, TypeError):
                    continue
        
        if bins is None:
            # Default bins: 20-29, 30-39, 40-49, 50-59, 60+
            bins = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 100)]
        
        distribution = {}
        for low, high in bins:
            count = sum(1 for age in ages if low <= age <= high)
            distribution[f"{low}-{high}"] = count
        
        return distribution
    
    def get_summary_statistics(self, column_name: str) -> Dict[str, float]:
        """Get comprehensive statistics for a numeric column."""
        return {
            'mean': self.mean(column_name),
            'median': self.median(column_name),
            'std': self.std(column_name),
            'min': self.min_value(column_name),
            'max': self.max_value(column_name),
            'count': len(self._get_column_values(column_name))
        }


def main():
    """Main function to demonstrate data generation and analysis."""
    processor = DataProcessor()
    
    # Generate employee data with salary and department info
    print("Generating employee data...")
    processor.generate_employee_data(100)
    processor.save_csv('output/employees.csv')
    print(f"Generated {len(processor.data)} employee records")
    
    # Display summary statistics for salary
    print("\n" + "="*50)
    print("SALARY STATISTICS")
    print("="*50)
    salary_stats = processor.get_summary_statistics('salary')
    for stat, value in salary_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Average salary per department
    print("\n" + "="*50)
    print("AVERAGE SALARY PER DEPARTMENT")
    print("="*50)
    avg_salaries = processor.avg_salary_per_department()
    for dept, avg_sal in sorted(avg_salaries.items()):
        print(f"{dept}: ${avg_sal:.2f}")
    
    # Employee count per department
    print("\n" + "="*50)
    print("EMPLOYEE COUNT PER DEPARTMENT")
    print("="*50)
    dept_counts = processor.employee_count_per_department()
    for dept, count in sorted(dept_counts.items()):
        print(f"{dept}: {count} employees")
    
    # Age distribution
    print("\n" + "="*50)
    print("AGE DISTRIBUTION")
    print("="*50)
    age_dist = processor.age_distribution()
    for age_range, count in age_dist.items():
        print(f"{age_range}: {count} employees")
    
    print("\nAll data generated and analyzed successfully!")


if __name__ == "__main__":
    main()
