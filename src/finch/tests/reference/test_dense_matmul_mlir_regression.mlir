Compiling MLIR code:
module {
  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_18: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
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
      %v_31 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_32 = arith.index_cast %v_31 : i64 to index
      %v_33 = arith.muli %v_32, %v_30 : index
      %v_34 = arith.addi %v_25, %v_33 : index
      scf.for %v_35 = %v_25 to %v_6 step %v_27 {
        %v_36 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_37 = arith.index_cast %v_36 : i64 to index
        %v_38 = arith.muli %v_37, %v_35 : index
        %v_39 = arith.addi %v_25, %v_38 : index
        %v_40 = llvm.extractvalue %_A_15[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_41 = arith.index_cast %v_40 : i64 to index
        %v_42 = arith.muli %v_41, %v_35 : index
        %v_43 = arith.addi %v_34, %v_42 : index
        scf.for %v_44 = %v_25 to %v_4 step %v_27 {
          %v_45 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_46 = arith.index_cast %v_45 : i64 to index
          %v_47 = arith.muli %v_46, %v_44 : index
          %v_48 = arith.addi %v_39, %v_47 : index
          %v_49 = arith.cmpi eq, %v_44, %v_30 : index
          scf.if %v_49 {
            %v_50 = memref.load %v_2[%v_43] : memref<?xf64>
            memref.store %v_50, %v_14[%v_48] : memref<?xf64>
          }
        }
      }
    }
    %v_51 = memref.dim %v_20, %v_25 : memref<?xf64>
    scf.for %v_52 = %v_25 to %v_51 step %v_27 {
      %v_53 = arith.constant 0.0 : f64
      memref.store %v_53, %v_20[%v_52] : memref<?xf64>
    }
    scf.for %v_54 = %v_25 to %v_10 step %v_27 {
      %v_55 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_56 = arith.index_cast %v_55 : i64 to index
      %v_57 = arith.muli %v_56, %v_54 : index
      %v_58 = arith.addi %v_25, %v_57 : index
      %v_59 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_60 = arith.index_cast %v_59 : i64 to index
      %v_61 = arith.muli %v_60, %v_54 : index
      %v_62 = arith.addi %v_25, %v_61 : index
      scf.for %v_63 = %v_25 to %v_18 step %v_27 {
        %v_64 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_65 = arith.index_cast %v_64 : i64 to index
        %v_66 = arith.muli %v_65, %v_63 : index
        %v_67 = arith.addi %v_25, %v_66 : index
        %v_68 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_69 = arith.index_cast %v_68 : i64 to index
        %v_70 = arith.muli %v_69, %v_63 : index
        %v_71 = arith.addi %v_58, %v_70 : index
        scf.for %v_72 = %v_25 to %v_12 step %v_27 {
          %v_73 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_74 = arith.index_cast %v_73 : i64 to index
          %v_75 = arith.muli %v_74, %v_72 : index
          %v_76 = arith.addi %v_67, %v_75 : index
          %v_77 = llvm.extractvalue %_A_16[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_78 = arith.index_cast %v_77 : i64 to index
          %v_79 = arith.muli %v_78, %v_72 : index
          %v_80 = arith.addi %v_62, %v_79 : index
          %v_81 = memref.load %v_20[%v_76] : memref<?xf64>
          %v_82 = memref.load %v_14[%v_71] : memref<?xf64>
          %v_83 = memref.load %v_8[%v_80] : memref<?xf64>
          %v_84 = arith.mulf %v_82, %v_83 : f64
          %v_85 = arith.addf %v_81, %v_84 : f64
          memref.store %v_85, %v_20[%v_76] : memref<?xf64>
        }
      }
    }
    %v_86 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_87 = llvm.insertvalue %_A_7, %v_86[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_87, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
