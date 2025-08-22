import json
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from virtual_sales_agent.graph import graph
from database.db_manager import DatabaseManager
from virtual_sales_agent.tools import (
    search_products,
    get_available_categories,
    check_order_status,
    search_products_recommendations
)

app = Flask(__name__)
CORS(app)

db_manager = DatabaseManager()

# Store active threads
active_threads = {}


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    customer_id = data.get('customer_id', 'default_customer')
    thread_id = data.get('thread_id')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Initialize thread state if needed
    if thread_id not in active_threads:
        thread_id = str(hash(customer_id + str(len(active_threads))))
        active_threads[thread_id] = {
            "messages": [],
            "user_info": json.dumps({"customer_id": customer_id})
        }
    
    # Add user message to thread
    active_threads[thread_id]["messages"].append({"role": "user", "content": message})
    
    # Process with LangGraph
    try:
        result = graph.invoke(active_threads[thread_id])
        
        # Extract the last assistant message
        response_text = None
        for msg in reversed(result["messages"]):
            if msg.get("role") == "assistant":
                response_text = msg.get("content")
                break
        
        if response_text is None:
            response_text = "I'm sorry, I couldn't process your request."
        
        # Update thread state
        active_threads[thread_id] = result
        
        # Get recommendations
        try:
            recommendations_result = search_products_recommendations(customer_id=customer_id)
            recommendations = recommendations_result.get('recommendations', [])
        except Exception as e:
            print(f"Error fetching recommendations: {str(e)}")
            recommendations = []
        
        return jsonify({
            "response": response_text,
            "thread_id": thread_id,
            "recommendations": recommendations
        })
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/products', methods=['GET'])
def get_products():
    query = request.args.get('query', '')
    category = request.args.get('category', '')
    min_price = float(request.args.get('min_price', 0))
    max_price = float(request.args.get('max_price', 1000))
    
    try:
        result = search_products(
            query=query if query else None,
            category=category if category else None,
            min_price=min_price,
            max_price=max_price
        )
        return jsonify(result.get('products', []))
    except Exception as e:
        print(f"Error fetching products: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        result = get_available_categories()
        return jsonify(result.get('categories', []))
    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/orders', methods=['GET'])
def get_orders():
    customer_id = request.args.get('customer_id')
    
    if not customer_id:
        return jsonify({"error": "Customer ID is required"}), 400
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get orders for the customer
            cursor.execute(
                """
                SELECT 
                    o.OrderId, 
                    o.OrderDate, 
                    o.Status
                FROM orders o
                WHERE o.CustomerId = ?
                ORDER BY o.OrderDate DESC
                """,
                (customer_id,)
            )
            
            orders = cursor.fetchall()
            formatted_orders = []
            
            for order in orders:
                # Get order items
                cursor.execute(
                    """
                    SELECT 
                        od.ProductId,
                        p.ProductName as name,
                        od.Quantity as quantity,
                        od.Price as price
                    FROM orders_details od
                    JOIN products p ON od.ProductId = p.ProductId
                    WHERE od.OrderId = ?
                    """,
                    (order['OrderId'],)
                )
                
                items = cursor.fetchall()
                
                formatted_orders.append({
                    "order_id": order['OrderId'],
                    "order_date": order['OrderDate'],
                    "status": order['Status'],
                    "items": [
                        {
                            "product_id": item['ProductId'],
                            "name": item['name'],
                            "quantity": item['quantity'],
                            "price": float(item['price'])
                        } for item in items
                    ]
                })
            
            return jsonify(formatted_orders)
    except Exception as e:
        print(f"Error fetching orders: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommendations', methods=['GET'])
def get_recommendations_route():
    customer_id = request.args.get('customer_id')
    
    if not customer_id:
        return jsonify({"error": "Customer ID is required"}), 400
    
    try:
        result = search_products_recommendations(customer_id=customer_id)
        return jsonify(result.get('recommendations', []))
    except Exception as e:
        print(f"Error fetching recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8502)