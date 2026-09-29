Compiling MLIR code:
module {
  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_4 = arith.index_cast %v_3 : i64 to index
    %v_5 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_6 = arith.index_cast %v_5 : i64 to index
    %v_7 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_8 = builtin.unrealized_conversion_cast %v_7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_9 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_12 = arith.index_cast %v_11 : i64 to index
    %v_13 = llvm.extractvalue %_A_18[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_15 = llvm.extractvalue %_A_18[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_18[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_20 = builtin.unrealized_conversion_cast %v_19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_21 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = arith.index_cast %v_21 : i64 to index
    %v_23 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = arith.constant 0 : index
    %v_26 = memref.dim %v_14, %v_25 : memref<?xf64>
    %v_27 = arith.constant 1 : index
    scf.for %v_28 = %v_25 to %v_26 step %v_27 {
      %v_29 = arith.constant 0.0 : f64
      memref.store %v_29, %v_14[%v_28] : memref<?xf64>
    }
    scf.for %v_30 = %v_25 to %v_4 step %v_27 {
      scf.for %v_31 = %v_25 to %v_6 step %v_27 {
        %v_32 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_33 = arith.index_cast %v_32 : i64 to index
        %v_34 = arith.muli %v_33, %v_31 : index
        %v_35 = arith.muli %v_30, %v_6 : index
        %v_36 = arith.addi %v_35, %v_31 : index
        scf.for %v_37 = %v_25 to %v_4 step %v_27 {
          %v_38 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_39 = arith.index_cast %v_38 : i64 to index
          %v_40 = arith.muli %v_39, %v_37 : index
          %v_41 = arith.addi %v_34, %v_40 : index
          %v_42 = arith.cmpi eq, %v_37, %v_30 : index
          scf.if %v_42 {
            %v_43 = memref.load %v_2[%v_36] : memref<?xf64>
            memref.store %v_43, %v_14[%v_41] : memref<?xf64>
          }
        }
      }
    }
    %v_44 = memref.dim %v_20, %v_25 : memref<?xf64>
    scf.for %v_45 = %v_25 to %v_44 step %v_27 {
      %v_46 = arith.constant 0.0 : f64
      memref.store %v_46, %v_20[%v_45] : memref<?xf64>
    }
    scf.for %v_47 = %v_25 to %v_10 step %v_27 {
      %v_48 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_49 = arith.index_cast %v_48 : i64 to index
      %v_50 = arith.muli %v_49, %v_47 : index
      scf.for %v_51 = %v_25 to %v_18 step %v_27 {
        %v_52 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_53 = arith.index_cast %v_52 : i64 to index
        %v_54 = arith.muli %v_53, %v_51 : index
        %v_55 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_56 = arith.index_cast %v_55 : i64 to index
        %v_57 = arith.muli %v_56, %v_51 : index
        %v_58 = arith.addi %v_50, %v_57 : index
        scf.for %v_59 = %v_25 to %v_12 step %v_27 {
          %v_60 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_61 = arith.index_cast %v_60 : i64 to index
          %v_62 = arith.muli %v_61, %v_59 : index
          %v_63 = arith.addi %v_54, %v_62 : index
          %v_64 = arith.muli %v_47, %v_12 : index
          %v_65 = arith.addi %v_64, %v_59 : index
          %v_66 = memref.load %v_20[%v_63] : memref<?xf64>
          %v_67 = memref.load %v_14[%v_58] : memref<?xf64>
          %v_68 = memref.load %v_8[%v_65] : memref<?xf64>
          %v_69 = arith.mulf %v_67, %v_68 : f64
          %v_70 = arith.addf %v_66, %v_69 : f64
          memref.store %v_70, %v_20[%v_63] : memref<?xf64>
        }
      }
    }
    %v_71 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_72 = llvm.insertvalue %_A_7, %v_71[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_72, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
