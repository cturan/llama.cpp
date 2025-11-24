#pragma once

#include <string>
#include <vector>
#include <functional>
#include <map>
#include <sstream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <regex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include "base64.hpp"
#include <random>
#include <cctype>
#include <climits>

// Inline tool callback function type
// Args are passed as strings, return value is string
using InlineToolCallback = std::function<std::string(const std::vector<std::string>&)>;

struct InlineTool {
    std::string name;
    InlineToolCallback callback;
    int num_args;
};

class InlineToolManager {
private:
    std::map<std::string, InlineTool> tools;

public:
    InlineToolManager() {
        // Register default tools
        
        // Addition
        register_tool("add", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) {
                return "ERROR: add() requires at least 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[add(5,10,15){==} → 30";
            }
            try {
                double sum = 0.0;
                for (size_t i = 0; i < args.size(); i++) {
                    try {
                        sum += std::stod(args[i]);
                    } catch (...) {
                        return "ERROR: add() argument " + std::to_string(i+1) + " is not a valid number: '" + args[i] + "'. Usage: [[add(5,10){==} → 15";
                    }
                }
                if (sum == std::floor(sum)) {
                    return std::to_string(static_cast<long long>(sum));
                }
                return std::to_string(sum);
            } catch (...) {
                return "ERROR: add() failed. Usage: [[add(5,10,15){==} → 30";
            }
        }, -1); // Variable number of args

        // Subtraction
        register_tool("sub", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: sub() requires exactly 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[sub(10,3){==} → 7";
            }
            try {
                double a = std::stod(args[0]);
                double b = std::stod(args[1]);
                double result = a - b;
                if (result == std::floor(result)) {
                    return std::to_string(static_cast<long long>(result));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: sub() arguments must be numbers. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[sub(10,3){==} → 7";
            }
        }, 2);

        // Division
        register_tool("div", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: div() requires exactly 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[div(10,2){==} → 5";
            }
            try {
                double a = std::stod(args[0]);
                double b = std::stod(args[1]);
                if (b == 0) {
                    return "ERROR: Division by zero (" + args[0] + "/0). Cannot divide by zero. Usage: [[div(10,2){==} → 5";
                }
                double result = a / b;
                if (result == std::floor(result)) {
                    return std::to_string(static_cast<long long>(result));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: div() arguments must be numbers. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[div(10,2){==} → 5";
            }
        }, 2);

        // Square Root
        register_tool("sqrt", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: sqrt() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[sqrt(25){==} → 5";
            }
            try {
                double a = std::stod(args[0]);
                if (a < 0) {
                    return "ERROR: sqrt() requires non-negative number, got " + args[0] + ". Cannot take square root of negative. Usage: [[sqrt(25){==} → 5";
                }
                double result = std::sqrt(a);
                if (result == std::floor(result)) {
                    return std::to_string(static_cast<long long>(result));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: sqrt() argument must be a number. Got: '" + args[0] + "'. Usage: [[sqrt(25){==} → 5";
            }
        }, 1);

        // Power
        register_tool("pow", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: pow() requires exactly 2 arguments (base, exponent), got " + std::to_string(args.size()) + ". Usage: [[pow(2,10){==} → 1024";
            }
            try {
                double base = std::stod(args[0]);
                double exp = std::stod(args[1]);
                double result = std::pow(base, exp);
                if (std::isnan(result) || std::isinf(result)) {
                    return "ERROR: pow(" + args[0] + "," + args[1] + ") resulted in invalid value. Usage: [[pow(2,10){==} → 1024";
                }
                if (result == std::floor(result)) {
                    return std::to_string(static_cast<long long>(result));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR";
            }
        }, 2);

        // Linear equation solver: ax + b = c
        register_tool("solve_linear", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 3) return "ERROR";
            try {
                double a = std::stod(args[0]);
                double b = std::stod(args[1]);
                double c = std::stod(args[2]);
                if (a == 0) return "ERROR: Not a linear equation";
                double x = (c - b) / a;
                if (x == std::floor(x)) {
                    return std::to_string(static_cast<long long>(x));
                }
                return std::to_string(x);
            } catch (...) {
                return "ERROR";
            }
        }, 3);

        // Quadratic equation solver: ax² + bx + c = 0
        register_tool("solve_quadratic", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 3) return "ERROR";
            try {
                double a = std::stod(args[0]);
                double b = std::stod(args[1]);
                double c = std::stod(args[2]);
                if (a == 0) return "ERROR: Not a quadratic equation";
                double discriminant = b*b - 4*a*c;
                if (discriminant < 0) return "ERROR: No real solutions";
                double sqrt_disc = std::sqrt(discriminant);
                double x1 = (-b + sqrt_disc) / (2*a);
                double x2 = (-b - sqrt_disc) / (2*a);
                std::string result = std::to_string(x1) + ", " + std::to_string(x2);
                return result;
            } catch (...) {
                return "ERROR";
            }
        }, 3);

        // Multiplication (existing)
        register_tool("multiply", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: multiply() requires exactly 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[multiply(7,8){==} → 56";
            }
            try {
                double a = std::stod(args[0]);
                double b = std::stod(args[1]);
                double result = a * b;
                if (std::isnan(result) || std::isinf(result)) {
                    return "ERROR: multiply(" + args[0] + "," + args[1] + ") resulted in invalid value. Usage: [[multiply(7,8){==} → 56";
                }
                if (result == std::floor(result)) {
                    return std::to_string(static_cast<long long>(result));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR";
            }
        }, 2);

        // ===== INTEGER ARITHMETIC =====
        
        // Factorial
        register_tool("factorial", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: factorial() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[factorial(5){==} → 120";
            }
            try {
                long long n = std::stoll(args[0]);
                if (n < 0) {
                    return "ERROR: factorial() requires non-negative integer, got " + args[0] + ". Usage: [[factorial(5){==} → 120";
                }
                if (n > 20) {
                    return "ERROR: factorial() input too large (>20), got " + args[0] + ". Would cause overflow. Usage: [[factorial(5){==} → 120";
                }
                long long result = 1;
                for (int i = 2; i <= n; i++) result *= i;
                return std::to_string(result);
            } catch (...) {
                return "ERROR: factorial() argument must be an integer. Got: '" + args[0] + "'. Usage: [[factorial(5){==} → 120";
            }
        }, 1);

        // GCD (variable args)
        register_tool("gcd", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) {
                return "ERROR: gcd() requires at least 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[gcd(48,18){==} → 6";
            }
            try {
                auto gcd_func = [](long long a, long long b) -> long long {
                    while (b) { long long t = b; b = a % b; a = t; }
                    return a;
                };
                long long result = std::stoll(args[0]);
                for (size_t i = 1; i < args.size(); i++) {
                    result = gcd_func(result, std::stoll(args[i]));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: gcd() arguments must be integers. Usage: [[gcd(48,18){==} → 6";
            }
        }, -1);

        // LCM (variable args)
        register_tool("lcm", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 2) {
                return "ERROR: lcm() requires at least 2 arguments, got " + std::to_string(args.size()) + ". Usage: [[lcm(12,15){==} → 60";
            }
            try {
                auto gcd_func = [](long long a, long long b) -> long long {
                    while (b) { long long t = b; b = a % b; a = t; }
                    return a;
                };
                auto lcm_func = [&](long long a, long long b) -> long long {
                    return (a / gcd_func(a, b)) * b;
                };
                long long result = std::stoll(args[0]);
                for (size_t i = 1; i < args.size(); i++) {
                    result = lcm_func(result, std::stoll(args[i]));
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: lcm() arguments must be integers. Usage: [[lcm(12,15){==} → 60";
            }
        }, -1);

        // Integer square root
        register_tool("isqrt", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: isqrt() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[isqrt(25){==} → 5";
            }
            try {
                long long n = std::stoll(args[0]);
                if (n < 0) {
                    return "ERROR: isqrt() requires non-negative integer, got " + args[0] + ". Usage: [[isqrt(25){==} → 5";
                }
                long long result = static_cast<long long>(std::sqrt(n));
                return std::to_string(result);
            } catch (...) { return "ERROR"; }
        }, 1);

        // Combinations: C(n,k) = n! / (k! * (n-k)!)
        register_tool("comb", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: comb() requires exactly 2 arguments (n,k), got " + std::to_string(args.size()) + ". Usage: [[comb(5,2){==} → 10";
            }
            try {
                long long n = std::stoll(args[0]);
                long long k = std::stoll(args[1]);
                if (n < 0 || k < 0) {
                    return "ERROR: comb() requires non-negative integers. Got n=" + args[0] + ", k=" + args[1] + ". Usage: [[comb(5,2){==} → 10";
                }
                if (k > n) {
                    return "ERROR: comb() requires k ≤ n. Got n=" + args[0] + ", k=" + args[1] + ". Usage: [[comb(5,2){==} → 10";
                }
                if (n > 30) {
                    return "ERROR: comb() input too large (>30), got n=" + args[0] + ". Would cause overflow. Usage: [[comb(5,2){==} → 10";
                }
                if (k > n - k) k = n - k;
                long long result = 1;
                for (int i = 0; i < k; i++) {
                    result = result * (n - i) / (i + 1);
                }
                return std::to_string(result);
            } catch (...) { return "ERROR"; }
        }, 2);

        // Permutations: P(n,k) = n! / (n-k)!
        register_tool("perm", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: perm() requires exactly 2 arguments (n,k), got " + std::to_string(args.size()) + ". Usage: [[perm(5,2){==} → 20";
            }
            try {
                long long n = std::stoll(args[0]);
                long long k = std::stoll(args[1]);
                if (n < 0 || k < 0) {
                    return "ERROR: perm() requires non-negative integers. Got n=" + args[0] + ", k=" + args[1] + ". Usage: [[perm(5,2){==} → 20";
                }
                if (k > n) {
                    return "ERROR: perm() requires k ≤ n. Got n=" + args[0] + ", k=" + args[1] + ". Usage: [[perm(5,2){==} → 20";
                }
                if (n > 20) {
                    return "ERROR: perm() input too large (>20), got n=" + args[0] + ". Would cause overflow. Usage: [[perm(5,2){==} → 20";
                }
                long long result = 1;
                for (int i = 0; i < k; i++) {
                    result *= (n - i);
                }
                return std::to_string(result);
            } catch (...) { return "ERROR"; }
        }, 2);

        // ===== FLOATING POINT MANIPULATION =====
        
        // Ceiling
        register_tool("ceil", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: ceil() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[ceil(3.2){==} → 4";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::ceil(x));
            } catch (...) {
                return "ERROR: ceil() argument must be a number. Got: '" + args[0] + "'. Usage: [[ceil(3.2){==} → 4";
            }
        }, 1);

        // Floor
        register_tool("floor", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: floor() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[floor(3.9){==} → 3";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::floor(x));
            } catch (...) {
                return "ERROR: floor() argument must be a number. Got: '" + args[0] + "'. Usage: [[floor(3.9){==} → 3";
            }
        }, 1);

        // Absolute value
        register_tool("fabs", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: fabs() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[fabs(-5.3){==} → 5.3";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::fabs(x));
            } catch (...) {
                return "ERROR: fabs() argument must be a number. Got: '" + args[0] + "'. Usage: [[fabs(-5.3){==} → 5.3";
            }
        }, 1);

        // Truncate
        register_tool("trunc", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: trunc() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[trunc(3.9){==} → 3";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::trunc(x));
            } catch (...) {
                return "ERROR: trunc() argument must be a number. Got: '" + args[0] + "'. Usage: [[trunc(3.9){==} → 3";
            }
        }, 1);

        // Floating point modulo
        register_tool("fmod", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: fmod() requires exactly 2 arguments (x,y), got " + std::to_string(args.size()) + ". Usage: [[fmod(10,3){==} → 1";
            }
            try {
                double x = std::stod(args[0]);
                double y = std::stod(args[1]);
                if (y == 0) {
                    return "ERROR: fmod() division by zero (mod by 0). Usage: [[fmod(10,3){==} → 1";
                }
                return std::to_string(std::fmod(x, y));
            } catch (...) {
                return "ERROR: fmod() arguments must be numbers. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[fmod(10,3){==} → 1";
            }
        }, 2);

        // ===== EXPONENTIAL & LOGARITHMIC =====
        
        // Cube root
        register_tool("cbrt", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: cbrt() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[cbrt(27){==} → 3";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::cbrt(x));
            } catch (...) {
                return "ERROR: cbrt() argument must be a number. Got: '" + args[0] + "'. Usage: [[cbrt(27){==} → 3";
            }
        }, 1);

        // e^x
        register_tool("exp", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: exp() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[exp(1){==} → 2.71828";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::exp(x);
                if (std::isinf(result)) {
                    return "ERROR: exp() result too large (overflow). Input: '" + args[0] + "'. Usage: [[exp(1){==} → 2.71828";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: exp() argument must be a number. Got: '" + args[0] + "'. Usage: [[exp(1){==} → 2.71828";
            }
        }, 1);

        // 2^x
        register_tool("exp2", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: exp2() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[exp2(3){==} → 8";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::exp2(x);
                if (std::isinf(result)) {
                    return "ERROR: exp2() result too large (overflow). Input: '" + args[0] + "'. Usage: [[exp2(3){==} → 8";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: exp2() argument must be a number. Got: '" + args[0] + "'. Usage: [[exp2(3){==} → 8";
            }
        }, 1);

        // e^x - 1
        register_tool("expm1", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: expm1() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[expm1(1){==} → 1.71828";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::expm1(x);
                if (std::isinf(result)) {
                    return "ERROR: expm1() result too large (overflow). Input: '" + args[0] + "'. Usage: [[expm1(1){==} → 1.71828";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: expm1() argument must be a number. Got: '" + args[0] + "'. Usage: [[expm1(1){==} → 1.71828";
            }
        }, 1);

        // Natural logarithm (ln)
        register_tool("log", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: log() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[log(2.71828){==} → 1";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= 0) {
                    return "ERROR: log() input must be positive (>0). Got: " + args[0] + ". Usage: [[log(2.71828){==} → 1";
                }
                return std::to_string(std::log(x));
            } catch (...) {
                return "ERROR: log() argument must be a number. Got: '" + args[0] + "'. Usage: [[log(2.71828){==} → 1";
            }
        }, 1);

        // Base-2 logarithm
        register_tool("log2", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: log2() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[log2(8){==} → 3";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= 0) {
                    return "ERROR: log2() input must be positive (>0). Got: " + args[0] + ". Usage: [[log2(8){==} → 3";
                }
                return std::to_string(std::log2(x));
            } catch (...) {
                return "ERROR: log2() argument must be a number. Got: '" + args[0] + "'. Usage: [[log2(8){==} → 3";
            }
        }, 1);

        // Base-10 logarithm
        register_tool("log10", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: log10() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[log10(100){==} → 2";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= 0) {
                    return "ERROR: log10() input must be positive (>0). Got: " + args[0] + ". Usage: [[log10(100){==} → 2";
                }
                return std::to_string(std::log10(x));
            } catch (...) {
                return "ERROR: log10() argument must be a number. Got: '" + args[0] + "'. Usage: [[log10(100){==} → 2";
            }
        }, 1);

        // ln(1+x)
        register_tool("log1p", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: log1p() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[log1p(1){==} → 0.693147";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= -1) {
                    return "ERROR: log1p() input must be > -1. Got: " + args[0] + ". Usage: [[log1p(1){==} → 0.693147";
                }
                return std::to_string(std::log1p(x));
            } catch (...) {
                return "ERROR: log1p() argument must be a number. Got: '" + args[0] + "'. Usage: [[log1p(1){==} → 0.693147";
            }
        }, 1);

        // ===== TRIGONOMETRIC FUNCTIONS =====
        
        // Sine
        register_tool("sin", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: sin() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[sin(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::sin(x));
            } catch (...) {
                return "ERROR: sin() argument must be a number (radians). Got: '" + args[0] + "'. Usage: [[sin(0){==} → 0";
            }
        }, 1);

        // Cosine
        register_tool("cos", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: cos() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[cos(0){==} → 1";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::cos(x));
            } catch (...) {
                return "ERROR: cos() argument must be a number (radians). Got: '" + args[0] + "'. Usage: [[cos(0){==} → 1";
            }
        }, 1);

        // Tangent
        register_tool("tan", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: tan() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[tan(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::tan(x);
                if (std::isinf(result)) {
                    return "ERROR: tan() undefined (asymptote). Input: '" + args[0] + "' (radians). Usage: [[tan(0){==} → 0";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: tan() argument must be a number (radians). Got: '" + args[0] + "'. Usage: [[tan(0){==} → 0";
            }
        }, 1);

        // Arcsine
        register_tool("asin", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: asin() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[asin(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                if (x < -1 || x > 1) {
                    return "ERROR: asin() input must be between -1 and 1. Got: " + args[0] + ". Usage: [[asin(0){==} → 0";
                }
                return std::to_string(std::asin(x));
            } catch (...) {
                return "ERROR: asin() argument must be a number. Got: '" + args[0] + "'. Usage: [[asin(0){==} → 0";
            }
        }, 1);

        // Arccosine
        register_tool("acos", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: acos() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[acos(1){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                if (x < -1 || x > 1) {
                    return "ERROR: acos() input must be between -1 and 1. Got: " + args[0] + ". Usage: [[acos(1){==} → 0";
                }
                return std::to_string(std::acos(x));
            } catch (...) {
                return "ERROR: acos() argument must be a number. Got: '" + args[0] + "'. Usage: [[acos(1){==} → 0";
            }
        }, 1);

        // Arctangent
        register_tool("atan", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: atan() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[atan(1){==} → 0.785398";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::atan(x));
            } catch (...) {
                return "ERROR: atan() argument must be a number. Got: '" + args[0] + "'. Usage: [[atan(1){==} → 0.785398";
            }
        }, 1);

        // Arctangent2 (y, x)
        register_tool("atan2", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: atan2() requires exactly 2 arguments (y,x), got " + std::to_string(args.size()) + ". Usage: [[atan2(1,1){==} → 0.785398";
            }
            try {
                double y = std::stod(args[0]);
                double x = std::stod(args[1]);
                return std::to_string(std::atan2(y, x));
            } catch (...) {
                return "ERROR: atan2() arguments must be numbers. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[atan2(1,1){==} → 0.785398";
            }
        }, 2);

        // Degrees to radians
        register_tool("radians", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: radians() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[radians(180){==} → 3.14159";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(x * M_PI / 180.0);
            } catch (...) {
                return "ERROR: radians() argument must be a number. Got: '" + args[0] + "'. Usage: [[radians(180){==} → 3.14159";
            }
        }, 1);

        // Radians to degrees
        register_tool("degrees", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: degrees() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[degrees(3.14159){==} → 180";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(x * 180.0 / M_PI);
            } catch (...) {
                return "ERROR: degrees() argument must be a number. Got: '" + args[0] + "'. Usage: [[degrees(3.14159){==} → 180";
            }
        }, 1);

        // ===== HYPERBOLIC FUNCTIONS =====
        
        // Hyperbolic sine
        register_tool("sinh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: sinh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[sinh(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::sinh(x);
                if (std::isinf(result)) {
                    return "ERROR: sinh() result too large (overflow). Input: '" + args[0] + "'. Usage: [[sinh(0){==} → 0";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: sinh() argument must be a number. Got: '" + args[0] + "'. Usage: [[sinh(0){==} → 0";
            }
        }, 1);

        // Hyperbolic cosine
        register_tool("cosh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: cosh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[cosh(0){==} → 1";
            }
            try {
                double x = std::stod(args[0]);
                double result = std::cosh(x);
                if (std::isinf(result)) {
                    return "ERROR: cosh() result too large (overflow). Input: '" + args[0] + "'. Usage: [[cosh(0){==} → 1";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: cosh() argument must be a number. Got: '" + args[0] + "'. Usage: [[cosh(0){==} → 1";
            }
        }, 1);

        // Hyperbolic tangent
        register_tool("tanh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: tanh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[tanh(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::tanh(x));
            } catch (...) {
                return "ERROR: tanh() argument must be a number. Got: '" + args[0] + "'. Usage: [[tanh(0){==} → 0";
            }
        }, 1);

        // Inverse hyperbolic sine
        register_tool("asinh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: asinh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[asinh(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::asinh(x));
            } catch (...) {
                return "ERROR: asinh() argument must be a number. Got: '" + args[0] + "'. Usage: [[asinh(0){==} → 0";
            }
        }, 1);

        // Inverse hyperbolic cosine
        register_tool("acosh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: acosh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[acosh(1){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                if (x < 1) {
                    return "ERROR: acosh() input must be >= 1. Got: " + args[0] + ". Usage: [[acosh(1){==} → 0";
                }
                return std::to_string(std::acosh(x));
            } catch (...) {
                return "ERROR: acosh() argument must be a number. Got: '" + args[0] + "'. Usage: [[acosh(1){==} → 0";
            }
        }, 1);

        // Inverse hyperbolic tangent
        register_tool("atanh", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: atanh() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[atanh(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= -1 || x >= 1) {
                    return "ERROR: atanh() input must be between -1 and 1. Got: " + args[0] + ". Usage: [[atanh(0){==} → 0";
                }
                return std::to_string(std::atanh(x));
            } catch (...) {
                return "ERROR: atanh() argument must be a number. Got: '" + args[0] + "'. Usage: [[atanh(0){==} → 0";
            }
        }, 1);

        // ===== SPECIAL FUNCTIONS =====
        
        // Error function
        register_tool("erf", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: erf() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[erf(0){==} → 0";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::erf(x));
            } catch (...) {
                return "ERROR: erf() argument must be a number. Got: '" + args[0] + "'. Usage: [[erf(0){==} → 0";
            }
        }, 1);

        // Complementary error function
        register_tool("erfc", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: erfc() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[erfc(0){==} → 1";
            }
            try {
                double x = std::stod(args[0]);
                return std::to_string(std::erfc(x));
            } catch (...) {
                return "ERROR: erfc() argument must be a number. Got: '" + args[0] + "'. Usage: [[erfc(0){==} → 1";
            }
        }, 1);

        // Gamma function
        register_tool("gamma", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: gamma() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[gamma(5){==} → 24";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= 0 && std::floor(x) == x) {
                    return "ERROR: gamma() undefined for non-positive integers. Got: " + args[0] + ". Usage: [[gamma(5){==} → 24";
                }
                double result = std::tgamma(x);
                if (std::isinf(result)) {
                    return "ERROR: gamma() result too large (overflow). Input: '" + args[0] + "'. Usage: [[gamma(5){==} → 24";
                }
                return std::to_string(result);
            } catch (...) {
                return "ERROR: gamma() argument must be a number. Got: '" + args[0] + "'. Usage: [[gamma(5){==} → 24";
            }
        }, 1);

        // Log gamma function
        register_tool("lgamma", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: lgamma() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[lgamma(5){==} → 3.17805";
            }
            try {
                double x = std::stod(args[0]);
                if (x <= 0 && std::floor(x) == x) {
                    return "ERROR: lgamma() undefined for non-positive integers. Got: " + args[0] + ". Usage: [[lgamma(5){==} → 3.17805";
                }
                return std::to_string(std::lgamma(x));
            } catch (...) {
                return "ERROR: lgamma() argument must be a number. Got: '" + args[0] + "'. Usage: [[lgamma(5){==} → 3.17805";
            }
        }, 1);

        // ===== MATHEMATICAL CONSTANTS =====
        
        // Pi constant
        register_tool("pi", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: pi() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[pi(){==} → 3.14159";
            }
            return std::to_string(M_PI);
        }, 0);

        // Euler's number
        register_tool("e", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: e() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[e(){==} → 2.71828";
            }
            return std::to_string(M_E);
        }, 0);

        // Tau (2*pi)
        register_tool("tau", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: tau() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[tau(){==} → 6.28318";
            }
            return std::to_string(2.0 * M_PI);
        }, 0);

        // ===== DATE & TIME FUNCTIONS =====
        
        // Get current Unix timestamp (seconds since epoch)
        register_tool("now", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: now() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[now(){==} → 1640995200";
            }
            try {
                auto now = std::chrono::system_clock::now();
                auto epoch = now.time_since_epoch();
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
                return std::to_string(seconds);
            } catch (...) {
                return "ERROR: now() failed to get current timestamp. Usage: [[now(){==} → 1640995200";
            }
        }, 0);

        // Get current date in YYYY-MM-DD format (UTC)
        register_tool("current_date", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: current_date() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[current_date(){==} → '2024-01-01'";
            }
            try {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm* now_tm = std::gmtime(&now_c);
                std::ostringstream oss;
                oss << std::put_time(now_tm, "%Y-%m-%d");
                return oss.str();
            } catch (...) {
                return "ERROR: current_date() failed to get current date. Usage: [[current_date(){==} → '2024-01-01'";
            }
        }, 0);

        // Get current time in HH:MM:SS format (UTC)
        register_tool("current_time", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: current_time() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[current_time(){==} → '12:30:45'";
            }
            try {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm* now_tm = std::gmtime(&now_c);
                std::ostringstream oss;
                oss << std::put_time(now_tm, "%H:%M:%S");
                return oss.str();
            } catch (...) {
                return "ERROR: current_time() failed to get current time. Usage: [[current_time(){==} → '12:30:45'";
            }
        }, 0);

        // Get current datetime in ISO 8601 format (UTC)
        register_tool("current_datetime", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: current_datetime() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[current_datetime(){==} → '2024-01-01 12:30:45 UTC'";
            }
            try {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm* now_tm = std::gmtime(&now_c);
                std::ostringstream oss;
                oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S UTC");
                return oss.str();
            } catch (...) {
                return "ERROR: current_datetime() failed to get current datetime. Usage: [[current_datetime(){==} → '2024-01-01 12:30:45 UTC'";
            }
        }, 0);

        // Get current datetime in local timezone
        register_tool("local_datetime", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: local_datetime() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[local_datetime(){==} → '2024-01-01 12:30:45 (UTC+0)'";
            }
            try {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm* local_tm = std::localtime(&now_c);
                std::ostringstream oss;
                oss << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S");
                
                // Get timezone offset
                std::tm* utc_tm = std::gmtime(&now_c);
                int offset_hours = local_tm->tm_hour - utc_tm->tm_hour;
                int offset_mins = local_tm->tm_min - utc_tm->tm_min;
                
                // Handle day boundary crossings
                if (local_tm->tm_mday != utc_tm->tm_mday) {
                    if (local_tm->tm_mday > utc_tm->tm_mday || 
                        (local_tm->tm_mday == 1 && utc_tm->tm_mday > 20)) {
                        offset_hours += 24;
                    } else {
                        offset_hours -= 24;
                    }
                }
                
                int total_offset = offset_hours;
                if (offset_mins != 0) {
                    total_offset = offset_hours >= 0 ? offset_hours : offset_hours;
                }
                
                oss << " (UTC";
                if (total_offset >= 0) oss << "+";
                oss << total_offset << ")";
                
                return oss.str();
            } catch (...) {
                return "ERROR: local_datetime() failed to get local datetime. Usage: [[local_datetime(){==} → '2024-01-01 12:30:45 (UTC+0)'";
            }
        }, 0);

        // Get system timezone offset (hours from UTC)
        register_tool("system_timezone_offset", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) {
                return "ERROR: system_timezone_offset() takes no arguments, got " + std::to_string(args.size()) + ". Usage: [[system_timezone_offset(){==} → '0'";
            }
            try {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm* local_tm = std::localtime(&now_c);
                std::tm* utc_tm = std::gmtime(&now_c);
                
                int offset_hours = local_tm->tm_hour - utc_tm->tm_hour;
                
                // Handle day boundary crossings
                if (local_tm->tm_mday != utc_tm->tm_mday) {
                    if (local_tm->tm_mday > utc_tm->tm_mday || 
                        (local_tm->tm_mday == 1 && utc_tm->tm_mday > 20)) {
                        offset_hours += 24;
                    } else {
                        offset_hours -= 24;
                    }
                }
                
                return std::to_string(offset_hours);
            } catch (...) {
                return "ERROR: system_timezone_offset() failed to get timezone offset. Usage: [[system_timezone_offset(){==} → '0'";
            }
        }, 0);

        // Add days to a Unix timestamp
        register_tool("add_days", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: add_days() requires exactly 2 arguments (timestamp, days), got " + std::to_string(args.size()) + ". Usage: [[add_days(1640995200,7){==} → 1641600000";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long days = std::stoll(args[1]);
                if (days < LLONG_MIN / 86400 || days > LLONG_MAX / 86400) {
                    return "ERROR: add_days() days value too large, got " + args[1] + ". Usage: [[add_days(1640995200,7){==} → 1641600000";
                }
                long long result = timestamp + (days * 86400LL); // 86400 seconds per day
                return std::to_string(result);
            } catch (const std::out_of_range&) {
                return "ERROR: add_days() arguments out of range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_days(1640995200,7){==} → 1641600000";
            } catch (const std::invalid_argument&) {
                return "ERROR: add_days() arguments must be numbers (timestamp, days). Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_days(1640995200,7){==} → 1641600000";
            }
        }, 2);

        // Add hours to a Unix timestamp
        register_tool("add_hours", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: add_hours() requires exactly 2 arguments (timestamp, hours), got " + std::to_string(args.size()) + ". Usage: [[add_hours(1640995200,24){==} → 1641081600";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long hours = std::stoll(args[1]);
                if (hours < LLONG_MIN / 3600 || hours > LLONG_MAX / 3600) {
                    return "ERROR: add_hours() hours value too large, got " + args[1] + ". Usage: [[add_hours(1640995200,24){==} → 1641081600";
                }
                long long result = timestamp + (hours * 3600LL); // 3600 seconds per hour
                return std::to_string(result);
            } catch (const std::out_of_range&) {
                return "ERROR: add_hours() arguments out of range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_hours(1640995200,24){==} → 1641081600";
            } catch (const std::invalid_argument&) {
                return "ERROR: add_hours() arguments must be numbers (timestamp, hours). Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_hours(1640995200,24){==} → 1641081600";
            }
        }, 2);

        // Add minutes to a Unix timestamp
        register_tool("add_minutes", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: add_minutes() requires exactly 2 arguments (timestamp, minutes), got " + std::to_string(args.size()) + ". Usage: [[add_minutes(1640995200,60){==} → 1640995260";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long minutes = std::stoll(args[1]);
                if (minutes < LLONG_MIN / 60 || minutes > LLONG_MAX / 60) {
                    return "ERROR: add_minutes() minutes value too large, got " + args[1] + ". Usage: [[add_minutes(1640995200,60){==} → 1640995260";
                }
                long long result = timestamp + (minutes * 60LL);
                return std::to_string(result);
            } catch (const std::out_of_range&) {
                return "ERROR: add_minutes() arguments out of range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_minutes(1640995200,60){==} → 1640995260";
            } catch (const std::invalid_argument&) {
                return "ERROR: add_minutes() arguments must be numbers (timestamp, minutes). Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[add_minutes(1640995200,60){==} → 1640995260";
            }
        }, 2);

        // Calculate difference in days between two timestamps
        register_tool("diff_days", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: diff_days() requires exactly 2 arguments (ts1, ts2), got " + std::to_string(args.size()) + ". Usage: [[diff_days(1640995200,1641600000){==} → 7";
            }
            try {
                long long ts1 = std::stoll(args[0]);
                long long ts2 = std::stoll(args[1]);
                long long diff = (ts2 - ts1) / 86400LL;
                return std::to_string(diff);
            } catch (...) {
                return "ERROR: diff_days() arguments must be timestamps. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[diff_days(1640995200,1641600000){==} → 7";
            }
        }, 2);

        // Calculate difference in hours between two timestamps
        register_tool("diff_hours", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: diff_hours() requires exactly 2 arguments (ts1, ts2), got " + std::to_string(args.size()) + ". Usage: [[diff_hours(1640995200,1641081600){==} → 24";
            }
            try {
                long long ts1 = std::stoll(args[0]);
                long long ts2 = std::stoll(args[1]);
                long long diff = (ts2 - ts1) / 3600LL;
                return std::to_string(diff);
            } catch (...) {
                return "ERROR: diff_hours() arguments must be timestamps. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[diff_hours(1640995200,1641081600){==} → 24";
            }
        }, 2);

        // Convert timestamp to readable date string
        register_tool("timestamp_to_date", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: timestamp_to_date() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[timestamp_to_date(1640995200){==} → '2022-01-01 00:00:00 UTC'";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                std::time_t time = static_cast<std::time_t>(timestamp);
                std::tm* tm_utc = std::gmtime(&time);
                std::ostringstream oss;
                oss << std::put_time(tm_utc, "%Y-%m-%d %H:%M:%S UTC");
                return oss.str();
            } catch (...) {
                return "ERROR: timestamp_to_date() argument must be a timestamp. Got: '" + args[0] + "'. Usage: [[timestamp_to_date(1640995200){==} → '2022-01-01 00:00:00 UTC'";
            }
        }, 1);

        // Get day of week from timestamp (0=Sunday, 6=Saturday)
        register_tool("day_of_week", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: day_of_week() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[day_of_week(1640995200){==} → 'Saturday'";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                std::time_t time = static_cast<std::time_t>(timestamp);
                std::tm* tm_utc = std::gmtime(&time);
                const char* days[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
                return days[tm_utc->tm_wday];
            } catch (...) {
                return "ERROR: day_of_week() argument must be a timestamp. Got: '" + args[0] + "'. Usage: [[day_of_week(1640995200){==} → 'Saturday'";
            }
        }, 1);

        // Convert time between timezones (offset in hours from UTC)
        // Args: timestamp, from_offset, to_offset
        // Example: timezone_convert(now, 3, 0) converts from UTC+3 (Istanbul) to UTC (London winter)
        register_tool("timezone_convert", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 3) {
                return "ERROR: timezone_convert() requires exactly 3 arguments (timestamp, from_offset, to_offset), got " + std::to_string(args.size()) + ". Usage: [[timezone_convert(1640995200,3,0){==} → 1640988000";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long from_offset = std::stoll(args[1]); // hours from UTC
                long long to_offset = std::stoll(args[2]);   // hours from UTC
                
                if (from_offset < LLONG_MIN / 3600 || from_offset > LLONG_MAX / 3600 ||
                    to_offset < LLONG_MIN / 3600 || to_offset > LLONG_MAX / 3600) {
                    return "ERROR: timezone_convert() offset values too large. Got: '" + args[1] + "', '" + args[2] + "'. Usage: [[timezone_convert(1640995200,3,0){==} → 1640988000";
                }
                
                // Adjust timestamp from source timezone to UTC, then to target timezone
                long long utc_timestamp = timestamp - (from_offset * 3600LL);
                long long result = utc_timestamp + (to_offset * 3600LL);
                
                return std::to_string(result);
            } catch (...) {
                return "ERROR: timezone_convert() arguments must be numbers (timestamp, from_offset, to_offset). Got: '" + args[0] + "', '" + args[1] + "', '" + args[2] + "'. Usage: [[timezone_convert(1640995200,3,0){==} → 1640988000";
            }
        }, 3);

        // Get hour from timestamp in a given timezone offset
        register_tool("get_hour_tz", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: get_hour_tz() requires exactly 2 arguments (timestamp, offset), got " + std::to_string(args.size()) + ". Usage: [[get_hour_tz(1640995200,3){==} → 3";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long tz_offset = std::stoll(args[1]); // hours from UTC
                
                if (tz_offset < LLONG_MIN / 3600 || tz_offset > LLONG_MAX / 3600) {
                    return "ERROR: get_hour_tz() timezone offset too large, got " + args[1] + ". Usage: [[get_hour_tz(1640995200,3){==} → 3";
                }
                
                long long adjusted = timestamp + (tz_offset * 3600LL);
                std::time_t time = static_cast<std::time_t>(adjusted);
                std::tm* tm_utc = std::gmtime(&time);
                
                return std::to_string(tm_utc->tm_hour);
            } catch (const std::out_of_range&) {
                return "ERROR: get_hour_tz() arguments out of range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[get_hour_tz(1640995200,3){==} → 3";
            } catch (const std::invalid_argument&) {
                return "ERROR: get_hour_tz() arguments must be numbers (timestamp, offset). Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[get_hour_tz(1640995200,3){==} → 3";
            }
        }, 2);

        // Get formatted datetime in a specific timezone
        register_tool("datetime_in_tz", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: datetime_in_tz() requires exactly 2 arguments (timestamp, offset), got " + std::to_string(args.size()) + ". Usage: [[datetime_in_tz(1640995200,3){==} → '2022-01-01 03:00:00 (UTC+3)'";
            }
            try {
                long long timestamp = std::stoll(args[0]);
                long long tz_offset = std::stoll(args[1]); // hours from UTC
                
                if (tz_offset < LLONG_MIN / 3600 || tz_offset > LLONG_MAX / 3600) {
                    return "ERROR: datetime_in_tz() timezone offset too large, got " + args[1] + ". Usage: [[datetime_in_tz(1640995200,3){==} → '2022-01-01 03:00:00 (UTC+3)'";
                }
                
                long long adjusted = timestamp + (tz_offset * 3600LL);
                std::time_t time = static_cast<std::time_t>(adjusted);
                std::tm* tm_utc = std::gmtime(&time);
                
                std::ostringstream oss;
                oss << std::put_time(tm_utc, "%Y-%m-%d %H:%M:%S (UTC");
                if (tz_offset >= 0) oss << "+";
                oss << tz_offset << ")";
                return oss.str();
            } catch (const std::out_of_range&) {
                return "ERROR: datetime_in_tz() arguments out of range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[datetime_in_tz(1640995200,3){==} → '2022-01-01 03:00:00 (UTC+3)'";
            } catch (const std::invalid_argument&) {
                return "ERROR: datetime_in_tz() arguments must be numbers (timestamp, offset). Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[datetime_in_tz(1640995200,3){==} → '2022-01-01 03:00:00 (UTC+3)'";
            }
        }, 2);

        // Create timestamp from date components (year, month, day, hour, minute, second)
        register_tool("make_timestamp", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 6) {
                return "ERROR: make_timestamp() requires exactly 6 arguments (y,m,d,h,m,s), got " + std::to_string(args.size()) + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
            }
            try {
                std::tm time_info = {};
                long long year = std::stoll(args[0]);
                long long month = std::stoll(args[1]);
                long long day = std::stoll(args[2]);
                long long hour = std::stoll(args[3]);
                long long minute = std::stoll(args[4]);
                long long second = std::stoll(args[5]);
                
                // Validate ranges
                if (year < 1970 || year > 2038) {
                    return "ERROR: make_timestamp() year must be between 1970-2038, got " + args[0] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                if (month < 1 || month > 12) {
                    return "ERROR: make_timestamp() month must be 1-12, got " + args[1] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                if (day < 1 || day > 31) {
                    return "ERROR: make_timestamp() day must be 1-31, got " + args[2] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                if (hour < 0 || hour > 23) {
                    return "ERROR: make_timestamp() hour must be 0-23, got " + args[3] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                if (minute < 0 || minute > 59) {
                    return "ERROR: make_timestamp() minute must be 0-59, got " + args[4] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                if (second < 0 || second > 59) {
                    return "ERROR: make_timestamp() second must be 0-59, got " + args[5] + ". Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
                }
                
                time_info.tm_year = static_cast<int>(year) - 1900;
                time_info.tm_mon = static_cast<int>(month) - 1;
                time_info.tm_mday = static_cast<int>(day);
                time_info.tm_hour = static_cast<int>(hour);
                time_info.tm_min = static_cast<int>(minute);
                time_info.tm_sec = static_cast<int>(second);
                
                std::time_t timestamp = timegm(&time_info);
                return std::to_string(timestamp);
            } catch (...) {
                return "ERROR: make_timestamp() arguments must be numbers (y,m,d,h,m,s). Got: '" + args[0] + "', '" + args[1] + "', '" + args[2] + "', '" + args[3] + "', '" + args[4] + "', '" + args[5] + "'. Usage: [[make_timestamp(2022,1,1,0,0,0){==} → 1640995200";
            }
        }, 6);

        // ===== STRING & ENCODING UTILITIES =====
        
        // Base64 encode
        register_tool("base64_encode", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: base64_encode() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[base64_encode('Hello'){==} → 'SGVsbG8='";
            }
            try {
                return base64::encode(args[0]);
            } catch (const std::exception& e) {
                return "ERROR: base64_encode() failed: " + std::string(e.what()) + ". Usage: [[base64_encode('Hello'){==} → 'SGVsbG8='";
            }
        }, 1);

        // Base64 decode
        register_tool("base64_decode", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: base64_decode() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[base64_decode('SGVsbG8='){==} → 'Hello'";
            }
            try {
                return base64::decode(args[0]);
            } catch (const base64_error& e) {
                return "ERROR: base64_decode() invalid input: " + std::string(e.what()) + ". Input must be valid base64. Usage: [[base64_decode('SGVsbG8='){==} → 'Hello'";
            } catch (const std::exception& e) {
                return "ERROR: base64_decode() failed: " + std::string(e.what()) + ". Usage: [[base64_decode('SGVsbG8='){==} → 'Hello'";
            }
        }, 1);

        // String length
        register_tool("strlen", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: strlen() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[strlen('Hello'){==} → 5";
            }
            return std::to_string(args[0].length());
        }, 1);

        // Substring (string, start, length)
        register_tool("substring", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 3) {
                return "ERROR: substring() requires exactly 3 arguments (string, start, length), got " + std::to_string(args.size()) + ". Usage: [[substring('Hello',0,3){==} → 'Hel'";
            }
            try {
                const std::string& str = args[0];
                size_t start = std::stoul(args[1]);
                size_t length = std::stoul(args[2]);
                if (start >= str.length()) {
                    return "ERROR: substring() start position " + args[1] + " is beyond string length " + std::to_string(str.length()) + ". Usage: [[substring('Hello',0,3){==} → 'Hel'";
                }
                return str.substr(start, length);
            } catch (...) {
                return "ERROR: substring() start and length must be numbers. Usage: [[substring('Hello',0,3){==} → 'Hel'";
            }
        }, 3);

        // String replace (string, old, new)
        register_tool("str_replace", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 3) {
                return "ERROR: str_replace() requires exactly 3 arguments (string, old, new), got " + std::to_string(args.size()) + ". Usage: [[str_replace('Hello','l','L'){==} → 'HeLLo'";
            }
            try {
                std::string result = args[0];
                const std::string& old_str = args[1];
                const std::string& new_str = args[2];
                
                size_t pos = 0;
                while ((pos = result.find(old_str, pos)) != std::string::npos) {
                    result.replace(pos, old_str.length(), new_str);
                    pos += new_str.length();
                }
                return result;
            } catch (...) { return "ERROR"; }
        }, 3);

        // Uppercase
        register_tool("uppercase", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: uppercase() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[uppercase('hello'){==} → 'HELLO'. Tip: Use quotes for strings with spaces.";
            }
            std::string result = args[0];
            std::transform(result.begin(), result.end(), result.begin(), ::toupper);
            return result;
        }, 1);

        // Lowercase
        register_tool("lowercase", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) {
                return "ERROR: lowercase() requires exactly 1 argument, got " + std::to_string(args.size()) + ". Usage: [[lowercase('HELLO'){==} → 'hello'. Tip: Use quotes for strings with spaces.";
            }
            std::string result = args[0];
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return result;
        }, 1);

        // String contains (haystack, needle) - returns "true" or "false"
        register_tool("str_contains", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: str_contains() requires exactly 2 arguments (string, substring), got " + std::to_string(args.size()) + ". Usage: [[str_contains('Hello, World','World'){==} → 'true'. Tip: Use quotes if string has commas/spaces.";
            }
            return args[0].find(args[1]) != std::string::npos ? "true" : "false";
        }, 2);

        // String starts with
        register_tool("str_startswith", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: str_startswith() requires exactly 2 arguments (string, prefix), got " + std::to_string(args.size()) + ". Usage: [[str_startswith('Hello','Hel'){==} → 'true'. Tip: Use quotes for strings.";
            }
            const std::string& str = args[0];
            const std::string& prefix = args[1];
            return str.size() >= prefix.size() && 
                   str.compare(0, prefix.size(), prefix) == 0 ? "true" : "false";
        }, 2);

        // String ends with
        register_tool("str_endswith", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: str_endswith() requires exactly 2 arguments (string, suffix), got " + std::to_string(args.size()) + ". Usage: [[str_endswith('Hello','lo'){==} → 'true'. Tip: Use quotes for strings.";
            }
            const std::string& str = args[0];
            const std::string& suffix = args[1];
            return str.size() >= suffix.size() && 
                   str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0 ? "true" : "false";
        }, 2);

        // URL encode
        register_tool("url_encode", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            try {
                std::string result;
                const std::string& input = args[0];
                
                for (unsigned char c : input) {
                    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                        result += c;
                    } else {
                        char hex[4];
                        snprintf(hex, sizeof(hex), "%%%02X", c);
                        result += hex;
                    }
                }
                return result;
            } catch (...) { return "ERROR"; }
        }, 1);

        // URL decode
        register_tool("url_decode", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            try {
                std::string result;
                const std::string& input = args[0];
                
                for (size_t i = 0; i < input.length(); i++) {
                    if (input[i] == '%' && i + 2 < input.length()) {
                        unsigned int value;
                        sscanf(input.substr(i + 1, 2).c_str(), "%x", &value);
                        result += static_cast<char>(value);
                        i += 2;
                    } else if (input[i] == '+') {
                        result += ' ';
                    } else {
                        result += input[i];
                    }
                }
                return result;
            } catch (...) { return "ERROR"; }
        }, 1);

        // Simple hash (djb2 algorithm) - for basic string hashing
        register_tool("hash_string", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            unsigned long hash = 5381;
            for (char c : args[0]) {
                hash = ((hash << 5) + hash) + static_cast<unsigned char>(c);
            }
            return std::to_string(hash);
        }, 1);

        // Random integer (min, max inclusive)
        register_tool("random_int", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) return "ERROR: random_int() requires exactly 2 arguments (min, max), got " + std::to_string(args.size()) + ". Usage: [[random_int(1,100){==} → 42";
            try {
                long long min = std::stoll(args[0]);
                long long max = std::stoll(args[1]);
                if (min > max) return "ERROR: random_int() min > max. Got min=" + args[0] + ", max=" + args[1] + ". Usage: [[random_int(1,100){==} → 42";
                
                static std::random_device rd;
                static std::mt19937_64 gen(rd());
                std::uniform_int_distribution<long long> dis(min, max);
                
                return std::to_string(dis(gen));
            } catch (const std::out_of_range&) {
                return "ERROR: random_int() arguments out of range. Values must be within 64-bit range. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[random_int(1,100){==} → 42";
            } catch (const std::invalid_argument&) {
                return "ERROR: random_int() arguments must be integers. Got: '" + args[0] + "', '" + args[1] + "'. Usage: [[random_int(1,100){==} → 42";
            } catch (...) {
                return "ERROR: random_int() failed with arguments: '" + args[0] + "', '" + args[1] + "'. Usage: [[random_int(1,100){==} → 42";
            }
        }, 2);

        // Generate UUID (simplified v4)
        register_tool("uuid", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) return "ERROR";
            try {
                static std::random_device rd;
                static std::mt19937 gen(rd());
                static std::uniform_int_distribution<> dis(0, 15);
                static std::uniform_int_distribution<> dis2(8, 11);
                
                std::stringstream ss;
                ss << std::hex;
                
                for (int i = 0; i < 8; i++) ss << dis(gen);
                ss << "-";
                for (int i = 0; i < 4; i++) ss << dis(gen);
                ss << "-4"; // version 4
                for (int i = 0; i < 3; i++) ss << dis(gen);
                ss << "-";
                ss << dis2(gen); // variant
                for (int i = 0; i < 3; i++) ss << dis(gen);
                ss << "-";
                for (int i = 0; i < 12; i++) ss << dis(gen);
                
                return ss.str();
            } catch (...) {
                return "ERROR: uuid() failed to generate unique ID. Usage: [[uuid(){==} → '123e4567-e89b-12d3-a456-426614174000'";
            }
        }, 0);

        // Regex match (returns "true" or "false")
        register_tool("regex_match", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: regex_match() requires exactly 2 arguments (string, pattern), got " + std::to_string(args.size()) + ". Usage: [[regex_match('hello','^h.*o$'){==} → 'true'";
            }
            try {
                std::regex pattern(args[1]);
                return std::regex_search(args[0], pattern) ? "true" : "false";
            } catch (const std::regex_error& e) {
                return "ERROR: regex_match() invalid pattern: '" + args[1] + "' - " + e.what() + ". Usage: [[regex_match('hello','^h.*o$'){==} → 'true'";
            } catch (...) {
                return "ERROR: regex_match() failed. Usage: [[regex_match('hello','^h.*o$'){==} → 'true'";
            }
        }, 2);

        // Split string by delimiter (returns count of parts)
        register_tool("str_split_count", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: str_split_count() requires exactly 2 arguments (string, delimiter), got " + std::to_string(args.size()) + ". Usage: [[str_split_count('a,b,c',','){==} → 3";
            }
            try {
                const std::string& str = args[0];
                const std::string& delim = args[1];
                
                if (str.empty()) return "0";
                if (delim.empty()) return "1";
                
                size_t count = 1;
                size_t pos = 0;
                while ((pos = str.find(delim, pos)) != std::string::npos) {
                    count++;
                    pos += delim.length();
                }
                return std::to_string(count);
            } catch (...) {
                return "ERROR: str_split_count() failed. Usage: [[str_split_count('a,b,c',','){==} → 3";
            }
        }, 2);

        // Count occurrences of a character/substring in a string
        register_tool("str_count_char", [](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) {
                return "ERROR: str_count_char() requires exactly 2 arguments (string, character), got " + std::to_string(args.size()) + ". Usage: [[str_count_char('Hello','l'){==} → 2. Tip: Use quotes for single chars like 'p'";
            }
            try {
                const std::string& str = args[0];
                const std::string& target = args[1];
                
                if (target.empty()) {
                    return "ERROR: str_count_char() target character cannot be empty. Usage: [[str_count_char('test','t'){==} → 2";
                }
                
                if (str.empty()) return "0";
                
                size_t count = 0;
                size_t pos = 0;
                while ((pos = str.find(target, pos)) != std::string::npos) {
                    count++;
                    pos += target.length();
                }
                return std::to_string(count);
            } catch (...) {
                return "ERROR: str_count_char() failed. Usage: [[str_count_char('Hello','l'){==} → 2. Remember: use quotes for single characters!";
            }
        }, 2);

        // MEMORY MANAGEMENT - Persistent user information storage
        // Memory file path (max 5KB)
        const std::string memory_file = "llama_memory.txt";
        const size_t MAX_MEMORY_SIZE = 5 * 1024; // 5KB

        // Read entire memory contents
        register_tool("memory_read", [memory_file](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) return "ERROR";
            try {
                std::ifstream file(memory_file);
                if (!file.is_open()) return "[Empty - No memory saved yet]";
                
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();
                
                if (content.empty()) return "[Empty - No memory saved yet]";
                return content;
            } catch (...) { return "ERROR"; }
        }, 0);

        // Save/overwrite entire memory (replaces all content)
        register_tool("memory_save", [memory_file, MAX_MEMORY_SIZE](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            try {
                const std::string& content = args[0];
                
                // Check size limit
                if (content.size() > MAX_MEMORY_SIZE) {
                    return "ERROR: Memory size exceeds 5KB limit";
                }
                
                std::ofstream file(memory_file, std::ios::trunc);
                if (!file.is_open()) return "ERROR: Cannot write to memory file";
                
                file << content;
                file.close();
                return "Memory saved successfully";
            } catch (...) { return "ERROR"; }
        }, 1);

        // Append text to memory
        register_tool("memory_append", [memory_file, MAX_MEMORY_SIZE](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            try {
                const std::string& new_content = args[0];
                
                // Read current content
                std::string current_content;
                std::ifstream infile(memory_file);
                if (infile.is_open()) {
                    std::stringstream buffer;
                    buffer << infile.rdbuf();
                    current_content = buffer.str();
                    infile.close();
                }
                
                // Add newline if current content doesn't end with one
                if (!current_content.empty() && current_content.back() != '\n') {
                    current_content += "\n";
                }
                
                current_content += new_content;
                
                // Check size limit
                if (current_content.size() > MAX_MEMORY_SIZE) {
                    return "ERROR: Memory would exceed 5KB limit";
                }
                
                // Write back
                std::ofstream outfile(memory_file, std::ios::trunc);
                if (!outfile.is_open()) return "ERROR: Cannot write to memory file";
                
                outfile << current_content;
                outfile.close();
                return "Content appended to memory";
            } catch (...) { return "ERROR"; }
        }, 1);

        // Clear all memory
        register_tool("memory_clear", [memory_file](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) return "ERROR";
            try {
                std::ofstream file(memory_file, std::ios::trunc);
                if (!file.is_open()) return "ERROR: Cannot write to memory file";
                file.close();
                return "Memory cleared successfully";
            } catch (...) { return "ERROR"; }
        }, 0);

        // Replace text in memory
        register_tool("memory_replace", [memory_file, MAX_MEMORY_SIZE](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 2) return "ERROR";
            try {
                const std::string& old_text = args[0];
                const std::string& new_text = args[1];
                
                // Read current content
                std::ifstream infile(memory_file);
                if (!infile.is_open()) return "ERROR: No memory file found";
                
                std::stringstream buffer;
                buffer << infile.rdbuf();
                std::string content = buffer.str();
                infile.close();
                
                // Find and replace
                size_t pos = content.find(old_text);
                if (pos == std::string::npos) {
                    return "ERROR: Text not found in memory";
                }
                
                content.replace(pos, old_text.length(), new_text);
                
                // Check size limit
                if (content.size() > MAX_MEMORY_SIZE) {
                    return "ERROR: Memory would exceed 5KB limit";
                }
                
                // Write back
                std::ofstream outfile(memory_file, std::ios::trunc);
                if (!outfile.is_open()) return "ERROR: Cannot write to memory file";
                
                outfile << content;
                outfile.close();
                return "Memory updated successfully";
            } catch (...) { return "ERROR"; }
        }, 2);

        // Remove specific line from memory (1-indexed)
        register_tool("memory_remove_line", [memory_file](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 1) return "ERROR";
            try {
                long long line_num = std::stoll(args[0]);
                if (line_num < 1) return "ERROR: Line number must be >= 1";
                if (line_num > INT_MAX) return "ERROR: Line number too large, got " + args[0];
                
                // Read all lines
                std::ifstream infile(memory_file);
                if (!infile.is_open()) return "ERROR: No memory file found";
                
                std::vector<std::string> lines;
                std::string line;
                while (std::getline(infile, line)) {
                    lines.push_back(line);
                }
                infile.close();
                
                // Check if line exists
                if (line_num > static_cast<int>(lines.size())) {
                    return "ERROR: Line number out of range";
                }
                
                // Remove line (convert from 1-indexed to 0-indexed)
                lines.erase(lines.begin() + (line_num - 1));
                
                // Write back
                std::ofstream outfile(memory_file, std::ios::trunc);
                if (!outfile.is_open()) return "ERROR: Cannot write to memory file";
                
                for (size_t i = 0; i < lines.size(); ++i) {
                    outfile << lines[i];
                    if (i < lines.size() - 1) outfile << "\n";
                }
                outfile.close();
                
                return "Line removed from memory";
            } catch (...) { return "ERROR"; }
        }, 1);

        // Get line count in memory
        register_tool("memory_line_count", [memory_file](const std::vector<std::string>& args) -> std::string {
            if (args.size() != 0) return "ERROR";
            try {
                std::ifstream file(memory_file);
                if (!file.is_open()) return "0";
                
                int count = 0;
                std::string line;
                while (std::getline(file, line)) {
                    count++;
                }
                file.close();
                
                return std::to_string(count);
            } catch (...) { return "ERROR"; }
        }, 0);
    }

    void register_tool(const std::string& name, InlineToolCallback callback, int num_args) {
        tools[name] = {name, callback, num_args};
    }

    // Returns empty string if no match or incomplete
    // Returns result string if matched and executed
    // Buffer should be the recent context ending with "{==}"
    // Format expected: [[tool_name(arg1, arg2){==}
    // Returns: [KERNEL_ANSWER: result]
    std::string check_and_execute(const std::string& buffer) {
        // Simple parser
        // Find the last "[["
        size_t open_pos = buffer.rfind("[[");
        if (open_pos == std::string::npos) return "";

        // Check if we have "{==}" at the end (or close to it, allowing for whitespace)
        size_t eq_pos = buffer.rfind("{==}");
        if (eq_pos == std::string::npos || eq_pos < open_pos) return "";

        // Extract the content between [[ and {==}
        std::string content = buffer.substr(open_pos + 2, eq_pos - (open_pos + 2));
        
        std::cerr << "[DEBUG] Tool content found: '" << content << "'" << std::endl;

        // Parse tool name and args
        // Content should be "name(arg1,arg2)"
        size_t paren_open = content.find("(");
        size_t paren_close = content.rfind(")");
        
        if (paren_open == std::string::npos || paren_close == std::string::npos || paren_close < paren_open) {
             std::cerr << "[DEBUG] Malformed tool call: Missing parentheses" << std::endl;
             return "";
        }

        std::string name = content.substr(0, paren_open);
        // trim name
        name.erase(0, name.find_first_not_of(" \t\n\r\f\v"));
        name.erase(name.find_last_not_of(" \t\n\r\f\v") + 1);

        std::cerr << "[DEBUG] Tool name: '" << name << "'" << std::endl;

        if (tools.find(name) == tools.end()) {
            std::cerr << "[DEBUG] Unknown tool: '" << name << "'" << std::endl;
            return "[KERNEL_ANSWER: ERROR - Unknown command '" + name + "'. Check available commands in system prompt.]";
        }

        std::string args_str = content.substr(paren_open + 1, paren_close - (paren_open + 1));
        std::vector<std::string> args;
        
        // Quote-aware argument parsing
        // Handles: strings with commas, quotes, multi-line, spaces
        std::string current_arg;
        char quote_char = 0; // 0 = not in quote, '\'' or '"' = in quote
        bool escaped = false;
        
        for (size_t i = 0; i < args_str.length(); i++) {
            char c = args_str[i];
            
            if (escaped) {
                current_arg += c;
                escaped = false;
                continue;
            }
            
            if (c == '\\') {
                escaped = true;
                continue;
            }
            
            // Handle quotes
            if (c == '\'' || c == '"') {
                if (quote_char == 0) {
                    // Start of quoted string - don't include the quote
                    quote_char = c;
                } else if (c == quote_char) {
                    // End of quoted string - don't include the quote
                    quote_char = 0;
                } else {
                    // Different quote inside quoted string
                    current_arg += c;
                }
                continue;
            }
            
            // Comma delimiter (only if not inside quotes)
            if (c == ',' && quote_char == 0) {
                // Trim and add argument
                size_t first = current_arg.find_first_not_of(" \t\n\r\f\v");
                if (first != std::string::npos) {
                    size_t last = current_arg.find_last_not_of(" \t\n\r\f\v");
                    args.push_back(current_arg.substr(first, (last - first + 1)));
                }
                current_arg.clear();
                continue;
            }
            
            // Regular character
            current_arg += c;
        }
        
        // Add last argument
        if (!current_arg.empty() || args_str.empty()) {
            size_t first = current_arg.find_first_not_of(" \t\n\r\f\v");
            if (first != std::string::npos) {
                size_t last = current_arg.find_last_not_of(" \t\n\r\f\v");
                args.push_back(current_arg.substr(first, (last - first + 1)));
            }
        }

        std::cerr << "[DEBUG] Args count: " << args.size() << std::endl;
        for(const auto& a : args) std::cerr << "[DEBUG] Arg: '" << a << "'" << std::endl;

        const auto& tool = tools[name];
        // For variable args (num_args = -1), check min 1 arg (or specific logic in callback)
        if (tool.num_args != -1 && args.size() != (size_t)tool.num_args) {
            std::cerr << "[DEBUG] Arg count mismatch. Expected " << tool.num_args << ", got " << args.size() << std::endl;
            return "[KERNEL_ANSWER: ERROR - Command '" + name + "' requires " + std::to_string(tool.num_args) + 
                   " arguments, but got " + std::to_string(args.size()) + ". Check command syntax.]";
        }

        // Execute
        std::string result = tool.callback(args);
        std::cerr << "[DEBUG] Tool result: " << result << std::endl;
        
        // Format: [KERNEL_ANSWER: result]
        // Inject a newline to clearly end the tool output line.
        return "[KERNEL_ANSWER: " + result + "]\n";
    }
};
