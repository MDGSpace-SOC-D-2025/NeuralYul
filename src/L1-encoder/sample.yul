object "MathContract" {
    code {
        // Constructor: Deploy the runtime logic
        datacopy(0, dataoffset("runtime"), datasize("runtime"))
        return(0, datasize("runtime"))
    }
    object "runtime" {
        code {
            // Function Dispatcher
            let selector := shr(224, calldataload(0))
            
            switch selector
            case 0x11223344 {
                // Read input, compute factorial, and return
                let n := calldataload(4)
                let res := compute_factorial(n)
                mstore(0, res)
                return(0, 32)
            }
            default {
                revert(0, 0)
            }

            // Isolated mathematical function
            function compute_factorial(n) -> result {
                result := 1
                // Replaced 'le(i, n)' with 'iszero(gt(i, n))'
                for { let i := 1 } iszero(gt(i, n)) { i := add(i, 1) }
                {
                    result := mul(result, i)
                }
            }
        }
    }
}
