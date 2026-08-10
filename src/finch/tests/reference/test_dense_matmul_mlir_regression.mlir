Compiling MLIR code:
module {
  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = arith.index_cast %v_3 : i64 to index
    %v_5 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = arith.index_cast %v_5 : i64 to index
    %v_7 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = builtin.unrealized_conversion_cast %v_7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_9 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = arith.index_cast %v_11 : i64 to index
    %v_13 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_15 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = arith.constant 0 : index
    %v_20 = memref.dim %v_14, %v_19 : memref<?xf64>
    %v_21 = arith.constant 1 : index
    scf.for %v_22 = %v_19 to %v_20 step %v_21 {
      %v_23 = arith.constant 0.0 : f64
      memref.store %v_23, %v_14[%v_22] : memref<?xf64>
    }
    scf.for %v_24 = %v_19 to %v_4 step %v_21 {
      %v_25 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_26 = arith.index_cast %v_25 : i64 to index
      %v_27 = arith.muli %v_26, %v_24 : index
      %v_28 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_29 = arith.index_cast %v_28 : i64 to index
      %v_30 = arith.muli %v_29, %v_24 : index
      scf.for %v_31 = %v_19 to %v_6 step %v_21 {
        %v_32 = llvm.extractvalue %_A_15[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_33 = arith.index_cast %v_32 : i64 to index
        %v_34 = arith.muli %v_33, %v_31 : index
        %v_35 = arith.addi %v_30, %v_34 : index
        %v_36 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_37 = arith.index_cast %v_36 : i64 to index
        %v_38 = arith.muli %v_37, %v_31 : index
        scf.for %v_39 = %v_19 to %v_12 step %v_21 {
          %v_40 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_41 = arith.index_cast %v_40 : i64 to index
          %v_42 = arith.muli %v_41, %v_39 : index
          %v_43 = arith.addi %v_27, %v_42 : index
          %v_44 = llvm.extractvalue %_A_16[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_45 = arith.index_cast %v_44 : i64 to index
          %v_46 = arith.muli %v_45, %v_39 : index
          %v_47 = arith.addi %v_38, %v_46 : index
          %v_48 = memref.load %v_14[%v_43] : memref<?xf64>
          %v_49 = memref.load %v_2[%v_35] : memref<?xf64>
          %v_50 = memref.load %v_8[%v_47] : memref<?xf64>
          %v_51 = arith.mulf %v_49, %v_50 : f64
          %v_52 = arith.addf %v_48, %v_51 : f64
          memref.store %v_52, %v_14[%v_43] : memref<?xf64>
        }
      }
    }
    %v_53 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_54 = llvm.insertvalue %_A_7, %v_53[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_54, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
