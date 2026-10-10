module {
  func.func @matmul(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.getelementptr %a[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_2 = llvm.load %v : !llvm.ptr -> !llvm.ptr
    %v_3 = llvm.getelementptr %a[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_4 = llvm.load %v_3 : !llvm.ptr -> i64
    %v_5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_6 = arith.constant 0 : i64
    %v_7 = arith.constant 1 : i64
    %v_8 = llvm.insertvalue %v_2, %v_5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_9 = llvm.insertvalue %v_2, %v_8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_10 = llvm.insertvalue %v_6, %v_9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_11 = llvm.insertvalue %v_4, %v_10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_12 = llvm.insertvalue %v_7, %v_11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = builtin.unrealized_conversion_cast %v_12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_15 = llvm.alloca %v_7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_14, %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_16 = llvm.getelementptr %b[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_17 = llvm.load %v_16 : !llvm.ptr -> !llvm.ptr
    %v_18 = llvm.getelementptr %b[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_19 = llvm.load %v_18 : !llvm.ptr -> i64
    %v_20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_21 = llvm.insertvalue %v_17, %v_20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_22 = llvm.insertvalue %v_17, %v_21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_23 = llvm.insertvalue %v_6, %v_22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_24 = llvm.insertvalue %v_19, %v_23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_25 = llvm.insertvalue %v_7, %v_24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_26 = builtin.unrealized_conversion_cast %v_25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_27 = builtin.unrealized_conversion_cast %v_26 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_28 = llvm.alloca %v_7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_27, %v_28 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_29 = llvm.getelementptr %c[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_30 = llvm.load %v_29 : !llvm.ptr -> !llvm.ptr
    %v_31 = llvm.getelementptr %c[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_32 = llvm.load %v_31 : !llvm.ptr -> i64
    %v_33 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_34 = llvm.insertvalue %v_30, %v_33[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_35 = llvm.insertvalue %v_30, %v_34[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_36 = llvm.insertvalue %v_6, %v_35[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_32, %v_36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_7, %v_37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = builtin.unrealized_conversion_cast %v_38 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_40 = builtin.unrealized_conversion_cast %v_39 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = llvm.alloca %v_7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_40, %v_41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_42 = arith.constant 0 : index
    %v_43 = arith.constant 2 : index
    %v_44 = arith.constant 1 : index
    scf.for %v_45 = %v_42 to %v_43 step %v_44 {
      %v_46 = arith.constant 4 : index
      scf.for %v_47 = %v_42 to %v_46 step %v_44 {
        %v_48 = arith.constant 3 : index
        scf.for %v_49 = %v_42 to %v_48 step %v_44 {
          %v_50 = llvm.load %v_41 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_51 = builtin.unrealized_conversion_cast %v_50 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_52 = llvm.load %v_41 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_53 = builtin.unrealized_conversion_cast %v_52 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_54 = arith.muli %v_45, %v_46 : index
          %v_55 = arith.addi %v_54, %v_47 : index
          %v_56 = memref.load %v_53[%v_55] : memref<?xf64>
          %v_57 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_58 = builtin.unrealized_conversion_cast %v_57 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_59 = arith.muli %v_45, %v_48 : index
          %v_60 = arith.addi %v_59, %v_49 : index
          %v_61 = memref.load %v_58[%v_60] : memref<?xf64>
          %v_62 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_63 = builtin.unrealized_conversion_cast %v_62 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_64 = arith.muli %v_49, %v_46 : index
          %v_65 = arith.addi %v_64, %v_47 : index
          %v_66 = memref.load %v_63[%v_65] : memref<?xf64>
          %v_67 = arith.mulf %v_61, %v_66 : f64
          %v_68 = arith.addf %v_56, %v_67 : f64
          %v_69 = arith.muli %v_45, %v_46 : index
          %v_70 = arith.addi %v_69, %v_47 : index
          memref.store %v_68, %v_51[%v_70] : memref<?xf64>
        }
      }
    }
    %v_71 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_72 = builtin.unrealized_conversion_cast %v_71 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_73 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_74 = builtin.unrealized_conversion_cast %v_73 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_75 = memref.dim %v_74, %v_42 : memref<?xf64>
    %v_76 = arith.index_cast %v_75 : index to i64
    %v_77 = llvm.getelementptr %a[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_76, %v_77 : i64, !llvm.ptr
    %v_78 = memref.extract_aligned_pointer_as_index %v_72 : memref<?xf64> -> index
    %v_79 = arith.index_cast %v_78 : index to i64
    %v_80 = llvm.inttoptr %v_79 : i64 to !llvm.ptr
    %v_81 = llvm.getelementptr %a[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_80, %v_81 : !llvm.ptr, !llvm.ptr
    %v_82 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_83 = builtin.unrealized_conversion_cast %v_82 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_84 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_85 = builtin.unrealized_conversion_cast %v_84 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_86 = memref.dim %v_85, %v_42 : memref<?xf64>
    %v_87 = arith.index_cast %v_86 : index to i64
    %v_88 = llvm.getelementptr %b[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_87, %v_88 : i64, !llvm.ptr
    %v_89 = memref.extract_aligned_pointer_as_index %v_83 : memref<?xf64> -> index
    %v_90 = arith.index_cast %v_89 : index to i64
    %v_91 = llvm.inttoptr %v_90 : i64 to !llvm.ptr
    %v_92 = llvm.getelementptr %b[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_91, %v_92 : !llvm.ptr, !llvm.ptr
    %v_93 = llvm.load %v_41 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_94 = builtin.unrealized_conversion_cast %v_93 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_95 = llvm.load %v_41 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_96 = builtin.unrealized_conversion_cast %v_95 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_97 = memref.dim %v_96, %v_42 : memref<?xf64>
    %v_98 = arith.index_cast %v_97 : index to i64
    %v_99 = llvm.getelementptr %c[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_98, %v_99 : i64, !llvm.ptr
    %v_100 = memref.extract_aligned_pointer_as_index %v_94 : memref<?xf64> -> index
    %v_101 = arith.index_cast %v_100 : index to i64
    %v_102 = llvm.inttoptr %v_101 : i64 to !llvm.ptr
    %v_103 = llvm.getelementptr %c[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_102, %v_103 : !llvm.ptr, !llvm.ptr
    func.return
  }
}
