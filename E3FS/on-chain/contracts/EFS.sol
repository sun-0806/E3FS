// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EFS {
    mapping(bytes32 => string) private data;

    event DataStored(bytes32 indexed key, string value);

    function storeData(bytes32 key, string memory value) public {
        data[key] = value;
        emit DataStored(key, value);
    }

    function batchStoreData(bytes32[] memory keys, string[] memory values) public {
        require(keys.length == values.length, "Keys and values length mismatch");
        for (uint256 i = 0; i < keys.length; i++) {
            data[keys[i]] = values[i];
            emit DataStored(keys[i], values[i]);
        }
    }

    function getData(bytes32 key) public view returns (string memory) {
        return data[key];
    }

    function exists(bytes32 key) public view returns (bool) {
        return bytes(data[key]).length > 0;
    }
}


